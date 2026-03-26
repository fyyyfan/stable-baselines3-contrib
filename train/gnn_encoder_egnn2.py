"""
基于 EGNN 思想的卫星–任务编码器（双向消息传递）
=====================================================

参考 `gnn_encoder.py` 的接口与观测约定，实现：

1. 双向消息传递使用不同边特征：
   - Task→Sat 方向使用 task_sat_dist（距离），让卫星感知任务的地理分布与负载压力。
   - Sat→Task 方向使用 task_sat_rate（传输速率），让任务根据带宽选择最优卫星。
2. 双向消息传递：Task→Sat (Sum 聚合感知负载) + Sat→Task (Attention 选择最优卫星)。
3. padding 节点（不足 I_max 的虚拟任务）在输出时特征置零，避免噪声。
4. `GNNFeaturesExtractor`：GNN readout 将全部卫星嵌入展平（保留个体差异），
   拼接任务嵌入与原始特征，经 MLP 映射到 `features_dim`（供 Critic 使用）。
5. 同时在 forward 中缓存结构化节点嵌入与原始观测
   （last_task_h / last_sat_h / last_edge_dist / last_edge_rate /
    last_task_features / last_sat_features），
   供 PointerNetMaskablePolicy 的 Actor 直接读取，实现逐 (task, sat) 对打分。

EGNN 相关形式参考 Satorras et al., E(n) Equivariant Graph Neural Networks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EGNNBlock(nn.Module):
    """
    双向 EGNN 消息传递块（两个方向使用不同边特征）。
    包含:
      1. Task -> Sat (感知拥塞): 使用 task_sat_dist 作为边特征，Sum 聚合
      2. Sat -> Task (分配资源): 使用 task_sat_rate 作为边特征，Attention 选择
    """
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 需被 num_heads 整除"
        
        self.hidden_dim = hidden_dim
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads
        
        in_edge = 2 * hidden_dim + edge_dim
        
        # ─── 1. Task -> Sat 网络 (边特征: task_sat_dist) ───
        self.msg_t2s = nn.Sequential(
            nn.Linear(in_edge, hidden_dim, bias=False),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.norm_sat = nn.LayerNorm(hidden_dim)

        # ─── 2. Sat -> Task 网络 (边特征: task_sat_rate) ───
        self.msg_s2t = nn.Linear(in_edge, hidden_dim, bias=False)
        self.attn_s2t = nn.Linear(in_edge, num_heads, bias=False)
        self.norm_task = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        sat_h: torch.Tensor,
        task_h: torch.Tensor,
        adj_task_to_sat: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sat_h:           (B, M, D)
            task_h:          (B, I, D)
            adj_task_to_sat: (B, I, M)
            edge_dist:       (B, I, M, edge_dim)  Task→Sat 方向使用的边特征 (距离)
            edge_rate:       (B, I, M, edge_dim)  Sat→Task 方向使用的边特征 (速率)
        """
        B, I, M = adj_task_to_sat.shape
        D = self.hidden_dim

        # ==========================================
        # 阶段 1: Task -> Sat (Sum 聚合, 使用 dist 边特征)
        # ==========================================
        ti = task_h.unsqueeze(2).expand(B, I, M, D)
        sj = sat_h.unsqueeze(1).expand(B, I, M, D)
        
        inp_t2s = torch.cat([ti, sj, edge_dist], dim=-1)    # (B, I, M, 2D + edge_dim)
        msg_t2s_out = self.msg_t2s(inp_t2s)                  # (B, I, M, D)
        
        mask_t2s = adj_task_to_sat.unsqueeze(-1)              # (B, I, M, 1)
        agg_sat = (msg_t2s_out * mask_t2s).sum(dim=1)         # (B, M, D)
        new_sat_h = self.norm_sat(sat_h + agg_sat)

        # ==========================================
        # 阶段 2: Sat -> Task (Softmax 注意力, 使用 rate 边特征)
        # ==========================================
        adj_s2t = adj_task_to_sat.transpose(1, 2)              # (B, M, I)
        edge_rate_s2t = edge_rate.transpose(1, 2)              # (B, M, I, edge_dim)
        
        si = new_sat_h.unsqueeze(2).expand(B, M, I, D)
        tj = task_h.unsqueeze(1).expand(B, M, I, D)
        
        inp_s2t = torch.cat([si, tj, edge_rate_s2t], dim=-1)  # (B, M, I, 2D + edge_dim)
        
        msg_s2t_out = self.msg_s2t(inp_s2t).view(B, M, I, self.num_heads, self.d_k)
        logits = self.attn_s2t(inp_s2t)                       # (B, M, I, H)
        
        mask_s2t = adj_s2t.unsqueeze(-1)                       # (B, M, I, 1)
        logits = logits.masked_fill(mask_s2t == 0, float("-inf"))
        
        alpha = torch.softmax(logits, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)
        
        agg_task = (alpha.unsqueeze(-1) * msg_s2t_out).sum(dim=1).reshape(B, I, D)
        new_task_h = self.norm_task(task_h + agg_task)

        return new_sat_h, new_task_h


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    EGNN 特征提取器，适配 MaskablePPO 的 MultiInputPolicy。

    拼接顺序 (Critic):
      [task_h_flat, sat_h_pooled,                          ← GNN readout
       task_features_raw, sat_features_raw,                ← 编码节点对应的原始特征
       其余观测键按字母序展平]                               ← 其余原始特征
      → MLP → features_dim

    Actor 缓存:
      last_task_h, last_sat_h,                             ← GNN 聚合后的节点嵌入
      last_edge_dist, last_edge_rate,                      ← 两类边特征矩阵
      last_task_features, last_sat_features                ← 原始节点特征
    """

    _NODE_KEYS = {"task_features", "sat_features"}

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        edge_dim: int = 1,
        dropout: float = 0.0,
    ):
        M = observation_space["sat_features"].shape[0]
        I_max = observation_space["task_features"].shape[0]
        d_sat = observation_space["sat_features"].shape[1]
        d_task = observation_space["task_features"].shape[1]

        self.M = M
        self.I_max = I_max
        self.d_sat = d_sat
        self.d_task = d_task
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # 结构化嵌入缓存（供 PointerNetMaskablePolicy Actor 读取）
        self.last_task_h: torch.Tensor | None = None          # (B, I, D)
        self.last_sat_h: torch.Tensor | None = None           # (B, M, D)
        self.last_edge_dist: torch.Tensor | None = None       # (B, I, M)
        self.last_edge_rate: torch.Tensor | None = None       # (B, I, M)
        self.last_task_features: torch.Tensor | None = None   # (B, I, d_task)
        self.last_sat_features: torch.Tensor | None = None    # (B, M, d_sat)

        # 原始特征拼接：先 task_features、sat_features，再其余键按字母序
        self._rest_keys = sorted(
            k for k in observation_space.spaces.keys() if k not in self._NODE_KEYS
        )
        raw_task_dim = int(np.prod(observation_space.spaces["task_features"].shape))
        raw_sat_dim = int(np.prod(observation_space.spaces["sat_features"].shape))
        rest_dim = sum(
            int(np.prod(observation_space.spaces[k].shape)) for k in self._rest_keys
        )

        # sat_h 展平（保留每颗卫星个体嵌入），替代原来的 mean pooling
        gnn_readout_dim = I_max * hidden_dim + M * hidden_dim
        fusion_in = gnn_readout_dim + raw_task_dim + raw_sat_dim + rest_dim
        hidden_f = max(features_dim, 256)

        super().__init__(observation_space, features_dim)

        # ─── 网络结构 ───
        self.sat_encoder = nn.Sequential(nn.Linear(d_sat, hidden_dim), nn.ELU())
        self.task_encoder = nn.Sequential(nn.Linear(d_task, hidden_dim), nn.ELU())

        self.gnn_layers = nn.ModuleList([
            EGNNBlock(hidden_dim, edge_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_f),
            nn.ELU(),
            nn.Linear(hidden_f, features_dim),
            nn.ELU(),
        )

        # ─── 正交初始化 ───
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self, observations: dict) -> torch.Tensor:
        # sat_feat = observations["sat_features"]
        # task_feat = observations["task_features"]
        # task_sat_adj = observations["task_sat_adj"]
        # task_sat_dist = observations["task_sat_dist"]
        # task_sat_rate = observations["task_sat_rate"]
        # 对输入进行数值清洗，防止环境中的极端值/无穷值传入网络导致 CUDA 异步错误。
        sat_feat = torch.nan_to_num(observations["sat_features"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        task_feat = torch.nan_to_num(observations["task_features"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        task_sat_adj = torch.nan_to_num(observations["task_sat_adj"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        task_sat_dist = torch.nan_to_num(observations["task_sat_dist"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        task_sat_rate = torch.nan_to_num(observations["task_sat_rate"].float(), nan=0.0, posinf=0.0, neginf=0.0)

        B = sat_feat.shape[0]

        # is_real_task 掩码 (task_features 最后一列)
        task_mask = task_feat[..., -1]  # (B, I_max), 1=真实 0=padding

        edge_dist = task_sat_dist.unsqueeze(-1)  # (B, I, M, 1)
        edge_rate = task_sat_rate.unsqueeze(-1)  # (B, I, M, 1)

        sat_h = self.sat_encoder(sat_feat)
        task_h = self.task_encoder(task_feat)

        for layer in self.gnn_layers:
            sat_h, task_h = layer(sat_h, task_h, task_sat_adj, edge_dist, edge_rate)

        # padding 节点特征置零，消除虚拟任务噪声
        task_h = task_h * task_mask.unsqueeze(-1)

        # ─── 缓存结构化嵌入与原始观测（供 PointerNetMaskablePolicy 的 Actor 读取）───
        self.last_task_h = task_h                    # (B, I, D)
        self.last_sat_h = sat_h                      # (B, M, D)
        self.last_edge_dist = task_sat_dist          # (B, I, M)
        self.last_edge_rate = task_sat_rate          # (B, I, M)
        self.last_task_features = task_feat          # (B, I, d_task)
        self.last_sat_features = sat_feat            # (B, M, d_sat)

        # ─── GNN Readout：sat_h 展平保留个体差异，替代 mean pooling ───
        gnn_vec = torch.cat(
            [task_h.reshape(B, -1), sat_h.reshape(B, -1)],
            dim=-1,
        )

        # ─── 原始特征拼接：task_features → sat_features → 其余键按字母序 ───
        raw_parts: list[torch.Tensor] = []
        raw_parts.append(task_feat.reshape(B, -1))
        raw_parts.append(sat_feat.reshape(B, -1))
        for key in self._rest_keys:
            raw_parts.append(observations[key].float().reshape(B, -1))

        raw_flat = torch.cat(raw_parts, dim=-1)
        combined = torch.cat([gnn_vec, raw_flat], dim=-1)

        return self.fusion(combined)