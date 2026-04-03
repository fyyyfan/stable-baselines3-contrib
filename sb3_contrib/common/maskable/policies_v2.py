"""
PointerNetMaskablePolicyV2 —— 纯节点特征 Pointer Network Actor + 结构化 Critic
=============================================================================

相比 policies_v1.py (PointerNetMaskablePolicy) 的改动：

1. Actor (PointerScorerV2):
   - 对每对 (task_i, sat_m) 拼接特征:
     [task_h_i, task_features_i, sat_h_m, sat_features_m]
   - 不使用边特征 (task_sat_dist, task_sat_rate) 作为打分输入
   - 经 MLP 输出标量得分，转为 MultiDiscrete logits

2. Critic (2个版本StructuredCritic, PoolingCritic):
   - 直接从 GNNFeaturesExtractor 缓存读取结构化节点嵌入与原始节点特征
   - 输入: [task_h_flat, sat_h_flat, task_features_flat, sat_features_flat]
   - 不使用边特征和其余观测键，保持与 Actor 特征集一致
   - 独立 MLP 输出标量价值

前提条件：
  - features_extractor 必须是 GNNFeaturesExtractor（来自 gnn_encoder_egnn2.py），
    其 forward 会将 last_task_h / last_sat_h / last_task_features / last_sat_features
    写入缓存。
  - 动作空间为 MultiDiscrete([B_actions] * I_max)。
  - 建议 net_arch=dict(pi=[], vf=[])，因为 Actor 和 Critic 均由自定义网络实现，
    MlpExtractor 仅作占位，不参与实际计算。

用法：
  policy_kwargs = dict(
      features_extractor_class=GNNFeaturesExtractor,
      features_extractor_kwargs=dict(...),
      share_features_extractor=False,
      net_arch=dict(pi=[], vf=[]),
      score_hidden=64,
      critic_hidden=256,
  )
  model = MaskablePPO(
      policy=PointerNetMaskablePolicyV2,
      ...
  )
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


# ======================================================================
# Actor: Pointer Network 打分器（纯节点特征）
# ======================================================================
class PointerScorerV2(nn.Module):
    """
    纯节点特征的 Pointer Network 动作打分网络。

    动作索引定义（与环境 MultiDiscrete 一致）：
      - 0:       本地执行（由 local_mlp 评估）
      - 1..M:    卸载到卫星 0..M-1（由 offload_mlp 评估）
      - M+1:     卸载到云端（由 offload_mlp 评估，云端为 sat_h 的第 M 个节点）

    本地动作打分：
      对每个 task_i 拼接 [task_h_i, task_features_i]，经 local_mlp 输出标量得分。

    卸载动作打分：
      对每对 (task_i, sat_m) 拼接特征:
        [task_h_i(D), task_features_i(d_task), sat_h_m(D), sat_features_m(d_sat)]
      经 offload_mlp 输出标量得分。不使用边特征 (dist, rate)。

    最终拼接顺序：[local_score, offload_scores]，reshape 为 (B, I_max * B_actions)。
    """

    def __init__(
        self,
        hidden_dim: int,
        d_task: int,
        d_sat: int,
        B_actions: int,
        I_max: int,
        score_hidden: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_task = d_task
        self.d_sat = d_sat
        self.B_actions = B_actions
        self.I_max = I_max

        # ── 本地执行打分: [task_h(D), task_feat(d_task)] → 标量 ──
        local_in_dim = hidden_dim + d_task
        self.local_mlp = nn.Sequential(
            nn.Linear(local_in_dim, score_hidden),
            nn.ELU(),
            nn.Linear(score_hidden, 1, bias=False),
        )

        # ── 卸载打分: [task_h(D), task_feat(d_task), sat_h(D), sat_feat(d_sat)] → 标量 ──
        offload_in_dim = 2 * hidden_dim + d_task + d_sat
        self.offload_mlp = nn.Sequential(
            nn.Linear(offload_in_dim, score_hidden),
            nn.ELU(),
            nn.Linear(score_hidden, 1, bias=False),
        )

        nn.init.orthogonal_(self.local_mlp[0].weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.local_mlp[2].weight, gain=0.01)
        nn.init.orthogonal_(self.offload_mlp[0].weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.offload_mlp[2].weight, gain=0.01)

    def forward(
        self,
        task_h: th.Tensor,          # (B, I, D)
        sat_h: th.Tensor,           # (B, M_sat, D)
        task_features: th.Tensor,   # (B, I, d_task)
        sat_features: th.Tensor,    # (B, M_sat, d_sat)
    ) -> th.Tensor:
        """
        返回 logits, shape (B, I_max * B_actions)。

        logits 按 [local(1), sat_0..sat_{M-1}(M), cloud(1)] 排列，
        对应动作索引 [0, 1, ..., M, M+1]。
        """
        B, I, D = task_h.shape
        M_sat = sat_h.shape[1]

        # ── 索引 0: 本地执行得分 ──
        local_inp = th.cat([task_h, task_features], dim=-1)            # (B, I, D+d_task)
        local_score = self.local_mlp(local_inp)                        # (B, I, 1)

        # ── 索引 1..M_sat: 卸载到卫星/云端的得分 ──
        ti = task_h.unsqueeze(2).expand(B, I, M_sat, D)
        tf = task_features.unsqueeze(2).expand(B, I, M_sat, self.d_task)
        sj = sat_h.unsqueeze(1).expand(B, I, M_sat, D)
        sf = sat_features.unsqueeze(1).expand(B, I, M_sat, self.d_sat)

        offload_inp = th.cat([ti, tf, sj, sf], dim=-1)                # (B, I, M, 2D+d_task+d_sat)
        offload_scores = self.offload_mlp(offload_inp).squeeze(-1)     # (B, I, M_sat)

        # ── 拼接: [local(1), offload(M_sat)] = B_actions 列 ──
        scores = th.cat([local_score, offload_scores], dim=-1)         # (B, I, 1+M_sat)

        assert scores.shape[-1] == self.B_actions, (
            f"scores dim {scores.shape[-1]} != B_actions {self.B_actions}"
        )

        return scores.reshape(B, I * self.B_actions)


# ======================================================================
# Actor: Attention-based Pointer Network 打分器（含边特征偏置）
# ======================================================================
class PointerScorerV2_Attention(nn.Module):
    """
    基于 Query-Key 注意力的 Pointer Network 动作打分网络。

    动作索引定义（与环境 MultiDiscrete 一致）：
      - 0:       本地执行（由 local_mlp 评估）
      - 1..M:    卸载到卫星 0..M-1（由 QK 注意力 + 边偏置评估）
      - M+1:     卸载到云端（同上，云端为 sat_h 的第 M 个节点）

    本地动作打分：
      对每个 task_i 拼接 [task_h_i, task_features_i]，经 local_mlp 输出标量得分。

    卸载动作打分：
      对任务 i 和卫星 j 分别拼接 [GNN 嵌入, 原始特征] 后做线性映射得到
      Query (B, I, H) 和 Key (B, M, H)，通过缩放点积注意力生成基础打分，
      再加上由成对边特征 (dist, rate) 线性投影得到的偏置项。

      score(i, j) = Q_i · K_j^T / sqrt(H)  +  edge_proj([dist_ij, rate_ij])

    最终拼接顺序：[local_score, offload_scores]，reshape 为 (B, I_max * B_actions)。
    """

    def __init__(
        self,
        hidden_dim: int,
        d_task: int,
        d_sat: int,
        B_actions: int,
        I_max: int,
        score_hidden: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_task = d_task
        self.d_sat = d_sat
        self.B_actions = B_actions
        self.I_max = I_max
        self.score_hidden = score_hidden

        # ── 本地执行打分: [task_h(D), task_feat(d_task)] → 标量 ──
        local_in_dim = hidden_dim + d_task
        self.local_mlp = nn.Sequential(
            nn.Linear(local_in_dim, score_hidden),
            nn.ELU(),
            nn.Linear(score_hidden, 1, bias=False),
        )

        # ── 卸载打分: QK 注意力 + 边偏置 ──
        self.query_norm = nn.LayerNorm(score_hidden)
        self.key_norm = nn.LayerNorm(score_hidden)

        self.query_proj = nn.Linear(hidden_dim + d_task, score_hidden)
        self.key_proj = nn.Linear(hidden_dim + d_sat, score_hidden)
        self.edge_proj = nn.Linear(2, 1, bias=False)

        nn.init.orthogonal_(self.local_mlp[0].weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.local_mlp[2].weight, gain=0.01)
        nn.init.orthogonal_(self.query_proj.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.key_proj.weight, gain=np.sqrt(2))
        with th.no_grad():
            self.edge_proj.weight.copy_(th.tensor([[-0.1, 0.1]]))

    def forward(
        self,
        task_h: th.Tensor,          # (B, I, D)
        sat_h: th.Tensor,           # (B, M_sat, D)
        task_features: th.Tensor,   # (B, I, d_task)
        sat_features: th.Tensor,    # (B, M_sat, d_sat)
        edge_dist: th.Tensor,       # (B, I, M_sat)
        edge_rate: th.Tensor,       # (B, I, M_sat)
    ) -> th.Tensor:
        """
        返回 logits, shape (B, I_max * B_actions)。

        logits 按 [local(1), sat_0..sat_{M-1}(M), cloud(1)] 排列，
        对应动作索引 [0, 1, ..., M, M+1]。
        """
        B, I, D = task_h.shape
        M_sat = sat_h.shape[1]

        # ── 索引 0: 本地执行得分 ──
        local_inp = th.cat([task_h, task_features], dim=-1)      # (B, I, D+d_task)
        local_score = self.local_mlp(local_inp)                   # (B, I, 1)

        # ── 索引 1..M_sat: 卸载到卫星/云端的得分 (QK attention + edge bias) ──
        task_combined = th.cat([task_h, task_features], dim=-1)   # (B, I, D+d_task)
        sat_combined = th.cat([sat_h, sat_features], dim=-1)     # (B, M, D+d_sat)

        queries = self.query_proj(task_combined)                  # (B, I, H)
        keys = self.key_proj(sat_combined)                        # (B, M, H)

        base_scores = th.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self.score_hidden)  # (B, I, M)

        edge_features = th.cat([edge_dist.unsqueeze(-1),
                                edge_rate.unsqueeze(-1)], dim=-1)  # (B, I, M, 2)
        edge_bias = self.edge_proj(edge_features).squeeze(-1)      # (B, I, M)

        offload_scores = base_scores + edge_bias                   # (B, I, M_sat)

        # ── 拼接: [local(1), offload(M_sat)] = B_actions 列 ──
        scores = th.cat([local_score, offload_scores], dim=-1)     # (B, I, 1+M_sat)

        assert scores.shape[-1] == self.B_actions, (
            f"scores dim {scores.shape[-1]} != B_actions {self.B_actions}"
        )

        return scores.reshape(B, I * self.B_actions)


# ======================================================================
# Critic: 结构化价值网络（纯节点特征）
# ======================================================================
class StructuredCritic(nn.Module):
    """
    结构化 Critic: 将 GNN 节点嵌入和原始节点特征拼接展平后, 经 MLP 输出 V(s).

    输入: [task_h_flat(I×D), sat_h_flat(M×D),
           task_features_flat(I×d_task), sat_features_flat(M×d_sat)]
    不使用边特征, 与 PointerScorerV2 保持一致。
    """

    def __init__(
        self,
        I_max: int,
        M: int,
        hidden_dim: int,
        d_task: int,
        d_sat: int,
        critic_hidden: int = 256,
    ):
        super().__init__()
        in_dim = I_max * hidden_dim + M * hidden_dim + I_max * d_task + M * d_sat

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, critic_hidden),
            nn.ELU(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.ELU(),
            nn.Linear(critic_hidden, 1),
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)
        # 输出层使用较小的增益
        nn.init.orthogonal_(self.mlp[-1].weight, gain=1.0)

    def forward(
        self,
        task_h: th.Tensor,          # (B, I, D)
        sat_h: th.Tensor,           # (B, M, D)
        task_features: th.Tensor,   # (B, I, d_task)
        sat_features: th.Tensor,    # (B, M, d_sat)
    ) -> th.Tensor:
        B = task_h.shape[0]
        combined = th.cat([
            task_h.reshape(B, -1),
            sat_h.reshape(B, -1),
            task_features.reshape(B, -1),
            sat_features.reshape(B, -1),
        ], dim=-1)
        return self.mlp(combined)


class PoolingCritic(nn.Module):
    """
    基于 mean-pooling 的全局 Critic。

    对 GNN 输出的节点嵌入和原始节点特征分别做 pooling，拼接为紧凑的
    全局状态向量后经 MLP 输出 V(s)。

    Pooling 策略:
      - task_h / task_features: masked mean pooling (仅对 is_real_task=1 的真实任务求均值)
      - sat_h / sat_features: mean pooling (所有卫星节点均为真实节点)

    输入维度 = 2 * hidden_dim + d_task + d_sat，与 I_max / M 无关。
    """

    def __init__(
        self,
        hidden_dim: int,
        d_task: int,
        d_sat: int,
        critic_hidden: int = 128,
    ):
        super().__init__()
        in_dim = 2 * hidden_dim + d_task + d_sat

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, critic_hidden),
            nn.ELU(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.ELU(),
            nn.Linear(critic_hidden, 1),
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)
        nn.init.orthogonal_(self.mlp[-1].weight, gain=1.0)

    def forward(
        self,
        task_h: th.Tensor,          # (B, I, D)
        sat_h: th.Tensor,           # (B, M, D)
        task_features: th.Tensor,   # (B, I, d_task) — 最后一列为 is_real_task
        sat_features: th.Tensor,    # (B, M, d_sat)
    ) -> th.Tensor:
        # task: masked mean pooling（padding 任务的 task_h 已由 GNN 置零）
        task_mask = task_features[..., -1:]                          # (B, I, 1)
        n_real = task_mask.sum(dim=1, keepdim=True).clamp(min=1)     # (B, 1, 1)
        task_h_pool = (task_h * task_mask).sum(dim=1) / n_real.squeeze(-1)       # (B, D)
        task_f_pool = (task_features * task_mask).sum(dim=1) / n_real.squeeze(-1) # (B, d_task)

        # sat: mean pooling
        sat_h_pool = sat_h.mean(dim=1)           # (B, D)
        sat_f_pool = sat_features.mean(dim=1)    # (B, d_sat)

        combined = th.cat([task_h_pool, sat_h_pool, task_f_pool, sat_f_pool], dim=-1)
        return self.mlp(combined)

# ======================================================================
# Policy: 纯节点特征版 Pointer Network + 结构化 Critic
# ======================================================================
class PointerNetMaskablePolicyV2(MaskableActorCriticPolicy):
    """
    纯节点特征版 Pointer Network MaskableActorCriticPolicy.

    Actor: PointerScorerV2
      输入 = [task_h, task_features, sat_h, sat_features] (来自 pi_features_extractor 缓存)
    Critic: StructuredCritic
      输入 = [task_h_flat, sat_h_flat, task_features_flat, sat_features_flat]
      (来自 vf_features_extractor 缓存)

    两者均不使用边特征 (dist, rate) 作为直接输入。
    GNN 内部消息传递仍然利用边特征, 但策略层只接收聚合后的节点嵌入与原始节点特征。

    相较于父类和 policies_new.py 的改动：
    1. Actor 打分去掉了 edge_dist / edge_rate。
    2. Critic 不再使用 MlpExtractor + value_net，
       改用 StructuredCritic 直接从 vf 缓存读取节点特征。
    3. 修正了 extract_features 的调用方式，
       避免 policies_new.py 中每次调用都冗余触发两个 extractor 的问题。
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = False,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        score_hidden: int = 64,
        critic_hidden: int = 128,
    ):
        if share_features_extractor:
            warnings.warn(
                "PointerNetMaskablePolicyV2 要求 share_features_extractor=False，已自动修正。",
                UserWarning,
            )
            share_features_extractor = False

        self._score_hidden = score_hidden
        self._critic_hidden = critic_hidden
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=False,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        构建网络：
        - Actor: PointerScorerV2（替代 action_net）
        - Critic: StructuredCritic（替代 MlpExtractor + value_net）
        - MlpExtractor / action_net / value_net 仅作占位，保持父类兼容性
        """
        self._build_mlp_extractor()

        extractor = self.pi_features_extractor
        hidden_dim = extractor.hidden_dim
        d_task = extractor.d_task
        d_sat = extractor.d_sat
        I_max = extractor.I_max
        M = extractor.M

        assert isinstance(self.action_space, spaces.MultiDiscrete), (
            "PointerNetMaskablePolicyV2 仅支持 MultiDiscrete 动作空间"
        )
        action_dims = list(self.action_space.nvec)
        assert len(set(action_dims)) == 1, "所有子动作维度须相同"
        B_actions = action_dims[0]

        # ── Actor: PointerScorerV2 ──
        self.pointer_scorer = PointerScorerV2(
            hidden_dim=hidden_dim,
            d_task=d_task,
            d_sat=d_sat,
            B_actions=B_actions,
            I_max=I_max,
            score_hidden=self._score_hidden,
        )

        # ── Critic: StructuredCritic ──
        # self.structured_critic = StructuredCritic(
        #     I_max=I_max,
        #     M=M,
        #     hidden_dim=hidden_dim,
        #     d_task=d_task,
        #     d_sat=d_sat,
        #     critic_hidden=self._critic_hidden,
        # )
        self.structured_critic = PoolingCritic(
            hidden_dim=hidden_dim,
            d_task=d_task,
            d_sat=d_sat,
            critic_hidden=self._critic_hidden,
        )


        # 占位件：父类序列化 / 其他工具方法可能检查这些属性
        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # 正交初始化（pointer_scorer / structured_critic 内部已完成）
        if self.ortho_init:
            module_gains = {
                self.pi_features_extractor: np.sqrt(2),
                self.vf_features_extractor: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                score_hidden=self._score_hidden,
                critic_hidden=self._critic_hidden,
            )
        )
        return data

    # ------------------------------------------------------------------
    # 内部工具：从 extractor 缓存中读取结构化嵌入
    # ------------------------------------------------------------------
    def _get_pointer_logits(self) -> th.Tensor:
        """从 pi_features_extractor 缓存生成 actor logits。"""
        ext = self.pi_features_extractor
        assert ext.last_task_h is not None, (
            "pi_features_extractor.forward 尚未被调用，缓存为空。"
        )
        return self.pointer_scorer(
            ext.last_task_h,
            ext.last_sat_h,
            ext.last_task_features,
            ext.last_sat_features,
        )

    def _get_structured_value(self) -> th.Tensor:
        """从 vf_features_extractor 缓存生成 critic value。"""
        ext = self.vf_features_extractor
        assert ext.last_task_h is not None, (
            "vf_features_extractor.forward 尚未被调用，缓存为空。"
        )
        return self.structured_critic(
            ext.last_task_h,
            ext.last_sat_h,
            ext.last_task_features,
            ext.last_sat_features,
        )

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor
    ) -> MaskableDistribution:
        """忽略 latent_pi，直接从缓存嵌入生成 logits。"""
        action_logits = self._get_pointer_logits()
        return self.action_dist.proba_distribution(action_logits=action_logits)

    # ------------------------------------------------------------------
    # 重写核心方法
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # 单次调用同时触发 pi / vf 两个 extractor，更新各自缓存
        self.extract_features(obs)

        distribution = self._get_action_dist_from_latent(th.empty(0))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        values = self._get_structured_value()
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        self.extract_features(obs)

        distribution = self._get_action_dist_from_latent(th.empty(0))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)

        values = self._get_structured_value()
        return values, log_prob, distribution.entropy()

    def get_distribution(
        self, obs: PyTorchObs, action_masks: np.ndarray | None = None
    ) -> MaskableDistribution:
        # 只需触发 pi extractor（跳过父类 MaskableActorCriticPolicy 的
        # extract_features，避免冗余调用 vf extractor）
        super(MaskableActorCriticPolicy, self).extract_features(
            obs, self.pi_features_extractor
        )
        distribution = self._get_action_dist_from_latent(th.empty(0))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        # 只需触发 vf extractor
        super(MaskableActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        return self._get_structured_value()


# ======================================================================
# Policy: Attention Pointer Network + PoolingCritic（含边特征偏置）
# ======================================================================
class PointerNetMaskablePolicyV2_Attention(MaskableActorCriticPolicy):
    """
    基于 Query-Key 注意力 + 边特征偏置的 Pointer Network MaskableActorCriticPolicy.

    Actor: PointerScorerV2_Attention
      对任务拼接特征做 Query 投影、卫星拼接特征做 Key 投影，缩放点积得到基础分数，
      再叠加由 (dist, rate) 线性投影的边偏置。

    Critic: PoolingCritic
      masked mean-pooling 聚合节点嵌入与原始特征后经 MLP 输出 V(s)。

    前提条件同 PointerNetMaskablePolicyV2，GNNFeaturesExtractor 需缓存
    last_task_h / last_sat_h / last_edge_dist / last_edge_rate /
    last_task_features / last_sat_features。
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = False,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        score_hidden: int = 64,
        critic_hidden: int = 256,
    ):
        if share_features_extractor:
            warnings.warn(
                "PointerNetMaskablePolicyV2_Attention 要求 share_features_extractor=False，已自动修正。",
                UserWarning,
            )
            share_features_extractor = False

        self._score_hidden = score_hidden
        self._critic_hidden = critic_hidden
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=False,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        extractor = self.pi_features_extractor
        hidden_dim = extractor.hidden_dim
        d_task = extractor.d_task
        d_sat = extractor.d_sat
        I_max = extractor.I_max

        assert isinstance(self.action_space, spaces.MultiDiscrete), (
            "PointerNetMaskablePolicyV2_Attention 仅支持 MultiDiscrete 动作空间"
        )
        action_dims = list(self.action_space.nvec)
        assert len(set(action_dims)) == 1, "所有子动作维度须相同"
        B_actions = action_dims[0]

        # ── Actor: PointerScorerV2_Attention ──
        self.pointer_scorer = PointerScorerV2_Attention(
            hidden_dim=hidden_dim,
            d_task=d_task,
            d_sat=d_sat,
            B_actions=B_actions,
            I_max=I_max,
            score_hidden=self._score_hidden,
        )

        # ── Critic: PoolingCritic ──
        self.structured_critic = PoolingCritic(
            hidden_dim=hidden_dim,
            d_task=d_task,
            d_sat=d_sat,
            critic_hidden=self._critic_hidden,
        )

        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.pi_features_extractor: np.sqrt(2),
                self.vf_features_extractor: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                score_hidden=self._score_hidden,
                critic_hidden=self._critic_hidden,
            )
        )
        return data

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _get_pointer_logits(self) -> th.Tensor:
        ext = self.pi_features_extractor
        assert ext.last_task_h is not None, (
            "pi_features_extractor.forward 尚未被调用，缓存为空。"
        )
        return self.pointer_scorer(
            ext.last_task_h,
            ext.last_sat_h,
            ext.last_task_features,
            ext.last_sat_features,
            ext.last_edge_dist,
            ext.last_edge_rate,
        )

    def _get_structured_value(self) -> th.Tensor:
        ext = self.vf_features_extractor
        assert ext.last_task_h is not None, (
            "vf_features_extractor.forward 尚未被调用，缓存为空。"
        )
        return self.structured_critic(
            ext.last_task_h,
            ext.last_sat_h,
            ext.last_task_features,
            ext.last_sat_features,
        )

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor
    ) -> MaskableDistribution:
        action_logits = self._get_pointer_logits()
        return self.action_dist.proba_distribution(action_logits=action_logits)

    # ------------------------------------------------------------------
    # 重写核心方法
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        self.extract_features(obs)

        distribution = self._get_action_dist_from_latent(th.empty(0))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        values = self._get_structured_value()
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        self.extract_features(obs)

        distribution = self._get_action_dist_from_latent(th.empty(0))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)

        values = self._get_structured_value()
        return values, log_prob, distribution.entropy()

    def get_distribution(
        self, obs: PyTorchObs, action_masks: np.ndarray | None = None
    ) -> MaskableDistribution:
        super(MaskableActorCriticPolicy, self).extract_features(
            obs, self.pi_features_extractor
        )
        distribution = self._get_action_dist_from_latent(th.empty(0))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        super(MaskableActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        return self._get_structured_value()
