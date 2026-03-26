"""
PointerNetMaskablePolicy —— Pointer Network Actor + Pooling Critic
====================================================================

动机：
  标准 MaskableActorCriticPolicy 中 Actor 的输入是一个全局 flat 向量，
  无法精确区分不同卫星的优劣。本策略直接利用 GNNFeaturesExtractor 缓存的
  结构化节点嵌入，为每个 (task_i, sat_m) 动作对构建特征向量并打分，
  类似 Pointer Network，使 Actor 天然感知每颗卫星的个体差异。

  Critic 采用 mean-pooling 对节点嵌入和原始节点特征进行全局聚合，
  生成紧凑的全局状态表示 V(s)，不依赖 I_max / M 的展平拼接。

前提条件：
  - features_extractor 必须是 GNNFeaturesExtractor（来自 gnn_encoder_egnn2.py），
    其 forward 会将 last_task_h / last_sat_h / last_edge_dist / last_edge_rate /
    last_task_features / last_sat_features 写入缓存。
  - 动作空间为 MultiDiscrete([B_actions] * I_max)。
  - 建议 net_arch=dict(pi=[], vf=[])，因为 Critic 由 PoolingCritic 实现，
    MlpExtractor 仅作占位。

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
      policy=PointerNetMaskablePolicy,
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


class PointerScorer(nn.Module):
    """
    Pointer Network 风格的动作打分网络。

    对每对 (task_i, sat_m) 拼接特征：
      [task_h_i, task_features_i, sat_h_m, sat_features_m, task_sat_dist_im, task_sat_rate_im]
    经 MLP 输出标量得分，最终 reshape 为 MultiDiscrete 所需的 logits。

    Args:
        hidden_dim:   GNN 节点嵌入维度 D
        d_task:       原始任务节点特征维度
        d_sat:        原始卫星节点特征维度
        B_actions:    每个子动作的选项数（卫星数 + 额外选项数）
        I_max:        最大任务数
        score_hidden: 打分 MLP 隐层宽度
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

        # 输入: [task_h(D), task_feat(d_task), sat_h(D), sat_feat(d_sat), dist(1), rate(1)]
        in_dim = 2 * hidden_dim + d_task + d_sat + 2

        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, score_hidden),
            nn.ELU(),
            nn.Linear(score_hidden, 1, bias=False),
        )

        self.extra_bias = nn.Parameter(th.zeros(B_actions))

        nn.init.orthogonal_(self.score_mlp[0].weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.score_mlp[2].weight, gain=0.01)

    def forward(
        self,
        task_h: th.Tensor,            # (B, I, D)        GNN 聚合后嵌入
        sat_h: th.Tensor,             # (B, M_sat, D)    GNN 聚合后嵌入
        task_features: th.Tensor,     # (B, I, d_task)   原始任务特征
        sat_features: th.Tensor,      # (B, M_sat, d_sat) 原始卫星特征
        edge_dist: th.Tensor,         # (B, I, M_sat)    距离边特征
        edge_rate: th.Tensor,         # (B, I, M_sat)    速率边特征
    ) -> th.Tensor:
        """
        返回 logits，shape (B, I_max * B_actions)，
        与 MaskableMultiCategoricalDistribution 的期望输入格式一致。
        """
        B, I, D = task_h.shape
        M_sat = sat_h.shape[1]
        extra = self.B_actions - M_sat

        # ── 构造所有 (task_i, sat_m) 对的输入特征 ──
        ti = task_h.unsqueeze(2).expand(B, I, M_sat, D)                        # (B,I,M,D)
        tf = task_features.unsqueeze(2).expand(B, I, M_sat, self.d_task)       # (B,I,M,d_task)
        sj = sat_h.unsqueeze(1).expand(B, I, M_sat, D)                        # (B,I,M,D)
        sf = sat_features.unsqueeze(1).expand(B, I, M_sat, self.d_sat)        # (B,I,M,d_sat)
        ed = edge_dist.unsqueeze(-1)                                           # (B,I,M,1)
        er = edge_rate.unsqueeze(-1)                                           # (B,I,M,1)

        inp = th.cat([ti, tf, sj, sf, ed, er], dim=-1)  # (B,I,M, 2D+d_task+d_sat+2)

        scores_sat = self.score_mlp(inp).squeeze(-1)     # (B, I, M_sat)

        if extra > 0:
            extra_scores = self.extra_bias[:extra].view(1, 1, extra).expand(B, I, extra)
            scores = th.cat([scores_sat, extra_scores], dim=-1)  # (B, I, B_actions)
        else:
            scores = scores_sat

        logits = scores.reshape(B, I * self.B_actions)
        return logits


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
        critic_hidden: int = 256,
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


class PointerNetMaskablePolicy(MaskableActorCriticPolicy):
    """
    Pointer Network Actor + Pooling Critic 的 MaskableActorCriticPolicy。

    相较于父类的改动：
    1. Actor 不再使用 MlpExtractor 的 latent_pi + Linear(action_net)，
       而是直接从 GNNFeaturesExtractor 的缓存中读取结构化节点嵌入，
       通过 PointerScorer 为每个 (task, sat) 对打分生成 logits。
    2. Critic 采用 PoolingCritic，对 GNN 节点嵌入做 mean-pooling 生成
       全局状态表示，经 MLP 输出 V(s)。不使用 MlpExtractor + value_net。
    3. 要求 share_features_extractor=False，因为 pi 和 vf 的输入形式不同。
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
                "PointerNetMaskablePolicy 要求 share_features_extractor=False，已自动修正。",
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
        - Actor: PointerScorer（替代 action_net）
        - Critic: PoolingCritic（替代 MlpExtractor + value_net）
        - MlpExtractor / action_net / value_net 仅作占位，保持父类兼容性
        """
        self._build_mlp_extractor()

        # ── Actor：PointerScorer ──
        extractor = self.pi_features_extractor
        hidden_dim = extractor.hidden_dim
        d_task = extractor.d_task
        d_sat = extractor.d_sat
        I_max = extractor.I_max

        assert isinstance(self.action_space, spaces.MultiDiscrete), (
            "PointerNetMaskablePolicy 仅支持 MultiDiscrete 动作空间"
        )
        action_dims = list(self.action_space.nvec)
        assert len(set(action_dims)) == 1, "所有子动作维度须相同"
        B_actions = action_dims[0]

        self.pointer_scorer = PointerScorer(
            hidden_dim=hidden_dim,
            d_task=d_task,
            d_sat=d_sat,
            B_actions=B_actions,
            I_max=I_max,
            score_hidden=self._score_hidden,
        )

        # ── Critic：PoolingCritic ──
        self.pooling_critic = PoolingCritic(
            hidden_dim=hidden_dim,
            d_task=d_task,
            d_sat=d_sat,
            critic_hidden=self._critic_hidden,
        )

        # 占位件：父类序列化等方法可能检查这些属性
        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # 正交初始化（pointer_scorer / pooling_critic 内部已完成初始化）
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

    # ------------------------------------------------------------------
    # 内部工具
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
            ext.last_edge_dist,
            ext.last_edge_rate,
        )

    def _get_pooling_value(self) -> th.Tensor:
        """从 vf_features_extractor 缓存生成 critic value。"""
        ext = self.vf_features_extractor
        assert ext.last_task_h is not None, (
            "vf_features_extractor.forward 尚未被调用，缓存为空。"
        )
        return self.pooling_critic(
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
        # 单次调用同时触发 pi / vf 两个 extractor，各运行一次，更新各自缓存
        self.extract_features(obs)

        distribution = self._get_action_dist_from_latent(th.empty(0))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        values = self._get_pooling_value()
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

        values = self._get_pooling_value()
        return values, log_prob, distribution.entropy()

    def get_distribution(
        self, obs: PyTorchObs, action_masks: np.ndarray | None = None
    ) -> MaskableDistribution:
        # 只需 pi extractor：绕过 MaskableActorCriticPolicy，直接调用 BasePolicy
        super(MaskableActorCriticPolicy, self).extract_features(
            obs, self.pi_features_extractor
        )
        distribution = self._get_action_dist_from_latent(th.empty(0))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        # 只需 vf extractor：绕过 MaskableActorCriticPolicy，直接调用 BasePolicy
        super(MaskableActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        return self._get_pooling_value()
