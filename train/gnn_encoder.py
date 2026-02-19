"""
GNN Features Extractor for MaskablePPO —— 异构 GAT 拓扑特征提取器
=====================================================================

实现了基于密集注意力机制的异构图注意力网络 (Heterogeneous GATv2)，
用于从卫星网络的异构二部图中提取拓扑特征。

架构:
  1. 节点编码器: 将 sat/task 原始特征映射到隐藏空间
  2. 多层异构 GAT 消息传递:
     - task → sat  (用户-卫星上行链路)
     - sat  → task (反向信息传播)
     - sat  → sat  (星间链路 ISL)
  3. Readout: 拼接所有 task embeddings + 全局 sat pooling + 任务数
     → MLP → 固定维度输出

适配 MaskablePPO:
  - 继承 BaseFeaturesExtractor
  - 输入: Dict 观测空间 (satellite_env_gnn.SatelliteGNNEnv 提供)
  - 输出: (batch, features_dim) 的固定维度特征向量

实现说明:
  采用密集矩阵运算实现 GAT 注意力，而非 PyG 的稀疏消息传递。
  这是因为:
    1. 卫星网络图规模较小 (M=4~10, I_max=6~10)，密集运算高效
    2. 天然支持批处理 (B, N, D)，无需 PyG Batch 转换
    3. 注意力机制与标准 GATv2 (Brody et al., 2021) 数学上等价
"""

from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# =====================================================================
# 1. 密集 GATv2 注意力层
# =====================================================================
class DenseGATv2Layer(nn.Module):
    """
    密集 GATv2 注意力层，支持边特征。

    注意力机制 (GATv2, Brody et al., 2021):
      e_ij = a^T · LeakyReLU(W_src·h_i + W_dst·h_j [+ W_edge·edge_ij])
      α_ij = softmax_{i ∈ N(j)} (e_ij)
      h_j' = Σ_{i ∈ N(j)} α_ij · W_val · h_i

    特点:
      - 使用密集邻接矩阵作为掩码，天然支持批处理
      - 支持可选的边特征 (edge_feat)
      - 多头注意力并行计算

    Args:
        in_dim:    输入节点特征维度
        out_dim:   输出节点特征维度 (需被 num_heads 整除)
        num_heads: 多头注意力头数
        edge_dim:  边特征维度 (0 表示不使用边特征)
        dropout:   注意力权重 Dropout
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_heads: int = 4, edge_dim: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        assert out_dim % num_heads == 0, \
            f"out_dim({out_dim}) 需被 num_heads({num_heads}) 整除"

        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        # ── 线性投影 ──
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)   # query
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)   # key
        self.W_val = nn.Linear(in_dim, out_dim, bias=False)   # value

        if edge_dim > 0:
            self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)

        # ── 注意力向量 (per head) ──
        self.attn = nn.Parameter(torch.empty(1, 1, 1, num_heads, self.d_k))
        nn.init.xavier_uniform_(self.attn.data.view(1, -1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                adj: torch.Tensor,
                edge_feat: torch.Tensor | None = None) -> torch.Tensor:
        """
        从 src 节点到 dst 节点的消息传递。

        Args:
            x_src:     (B, N_src, d_in) 源节点特征
            x_dst:     (B, N_dst, d_in) 目标节点特征
            adj:       (B, N_src, N_dst) 邻接掩码 (1=有边, 0=无边)
            edge_feat: (B, N_src, N_dst, edge_dim) 边特征 (可选)

        Returns:
            out: (B, N_dst, d_out) 更新后的目标节点特征
        """
        B, N_src, _ = x_src.shape
        N_dst = x_dst.shape[1]
        H, D = self.num_heads, self.d_k

        # ── 线性投影 → 多头 ──
        q = self.W_src(x_src).view(B, N_src, H, D)     # (B, N_src, H, D)
        k = self.W_dst(x_dst).view(B, N_dst, H, D)     # (B, N_dst, H, D)
        v = self.W_val(x_src).view(B, N_src, H, D)     # (B, N_src, H, D)

        # ── 展开为成对 (pairwise) 计算 ──
        q_exp = q.unsqueeze(2).expand(-1, -1, N_dst, -1, -1)  # (B, N_src, N_dst, H, D)
        k_exp = k.unsqueeze(1).expand(-1, N_src, -1, -1, -1)  # (B, N_src, N_dst, H, D)

        attn_input = q_exp + k_exp  # GATv2: 先加后非线性

        if edge_feat is not None and self.edge_dim > 0:
            e_proj = self.W_edge(edge_feat).view(B, N_src, N_dst, H, D)
            attn_input = attn_input + e_proj

        attn_input = self.leaky_relu(attn_input)

        # ── 注意力分数 ──
        e = (attn_input * self.attn).sum(dim=-1)  # (B, N_src, N_dst, H)

        # ── 掩码非边 (设为 -inf 使 softmax 后为 0) ──
        mask = adj.unsqueeze(-1)  # (B, N_src, N_dst, 1)
        e = e.masked_fill(mask == 0, float('-inf'))

        # ── Softmax: 对每个 dst 节点 j，在所有 src 邻居 i 上归一化 ──
        alpha = torch.softmax(e, dim=1)  # (B, N_src, N_dst, H)
        # 处理孤立节点 (全 -inf → NaN)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # ── 加权聚合 ──
        v_exp = v.unsqueeze(2).expand(-1, -1, N_dst, -1, -1)  # (B, N_src, N_dst, H, D)
        out = (alpha.unsqueeze(-1) * v_exp).sum(dim=1)         # (B, N_dst, H, D)

        return out.reshape(B, N_dst, H * D)


# =====================================================================
# 2. 异构 GAT 消息传递块
# =====================================================================
class HeteroGATBlock(nn.Module):
    """
    异构 GAT 消息传递块，包含三种边类型:

      1. task → sat  (用户-卫星上行链路可见性)
      2. sat  → task (反向信息传播)
      3. sat  → sat  (星间链路 ISL)

    每种边类型使用独立的 DenseGATv2Layer。
    包含残差连接和 LayerNorm 以稳定训练。

    Args:
        hidden_dim: 隐藏层维度
        num_heads:  多头注意力头数
        edge_dim:   边特征维度
        dropout:    Dropout 概率
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 edge_dim: int = 2, dropout: float = 0.0):
        super().__init__()

        self.gat_task_to_sat = DenseGATv2Layer(
            hidden_dim, hidden_dim, num_heads, edge_dim, dropout
        )
        self.gat_sat_to_task = DenseGATv2Layer(
            hidden_dim, hidden_dim, num_heads, edge_dim, dropout
        )
        self.gat_sat_to_sat = DenseGATv2Layer(
            hidden_dim, hidden_dim, num_heads, edge_dim, dropout
        )

        self.norm_task = nn.LayerNorm(hidden_dim)
        self.norm_sat = nn.LayerNorm(hidden_dim)
        self.act = nn.ELU()

    def forward(
        self,
        task_h: torch.Tensor,
        sat_h: torch.Tensor,
        task_sat_adj: torch.Tensor,
        task_sat_feat: torch.Tensor,
        sat_sat_adj: torch.Tensor,
        sat_sat_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        一步异构消息传递。

        Args:
            task_h:        (B, I_max, d) 任务节点嵌入
            sat_h:         (B, M, d)     卫星节点嵌入
            task_sat_adj:  (B, I_max, M) 任务→卫星邻接矩阵
            task_sat_feat: (B, I_max, M, edge_dim) 任务→卫星边特征
            sat_sat_adj:   (B, M, M)     卫星→卫星邻接矩阵
            sat_sat_feat:  (B, M, M, edge_dim) 卫星→卫星边特征

        Returns:
            new_task_h: (B, I_max, d) 更新后的任务嵌入
            new_sat_h:  (B, M, d)     更新后的卫星嵌入
        """
        # ── task → sat: 卫星聚合来自可见任务的信息 ──
        msg_task_to_sat = self.gat_task_to_sat(
            task_h, sat_h, task_sat_adj, task_sat_feat
        )

        # ── sat → task: 任务聚合来自可见卫星的信息 ──
        # 转置邻接矩阵和边特征
        adj_rev = task_sat_adj.transpose(1, 2)           # (B, M, I_max)
        feat_rev = task_sat_feat.permute(0, 2, 1, 3)     # (B, M, I_max, edge_dim)
        msg_sat_to_task = self.gat_sat_to_task(
            sat_h, task_h, adj_rev, feat_rev
        )

        # ── sat → sat: 卫星聚合来自 ISL 邻居的信息 ──
        msg_sat_to_sat = self.gat_sat_to_sat(
            sat_h, sat_h, sat_sat_adj, sat_sat_feat
        )

        # ── 残差连接 + LayerNorm ──
        new_sat_h = self.norm_sat(
            sat_h + self.act(msg_task_to_sat + msg_sat_to_sat)
        )
        new_task_h = self.norm_task(
            task_h + self.act(msg_sat_to_task)
        )

        return new_task_h, new_sat_h


# =====================================================================
# 3. GNN 特征提取器 (适配 MaskablePPO)
# =====================================================================
class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    异构 GAT 特征提取器，适配 MaskablePPO 的 MultiInputPolicy。

    将 Dict 观测空间（包含节点特征、邻接矩阵、边特征）
    编码为固定维度的特征向量，送入 MaskablePPO 的 Actor-Critic 网络。

    输入 (Dict 观测空间，由 SatelliteGNNEnv 提供):
        sat_features:  (M, 4)        卫星节点特征
        task_features: (I_max, 4)    任务节点特征
        task_sat_adj:  (I_max, M)    任务→卫星邻接矩阵
        task_sat_dist: (I_max, M)    归一化星地距离
        task_sat_rate: (I_max, M)    估计上行传输速率
        sat_sat_adj:   (M, M)        卫星间 ISL 邻接矩阵
        sat_sat_dist:  (M, M)        归一化星间距离
        sat_sat_rate:  (M, M)        ISL 数据传输速率
        num_tasks:     (1,)          实际任务数 (归一化)

    输出:
        features: (B, features_dim)  固定维度特征向量

    Args:
        observation_space: gym.spaces.Dict 观测空间
        features_dim:      输出特征维度 (默认 128)
        hidden_dim:        GNN 隐藏层维度 (默认 64)
        num_gnn_layers:    GAT 消息传递层数 (默认 2)
        num_heads:         多头注意力头数 (默认 4, 需整除 hidden_dim)
        edge_dim:          边特征维度 (默认 2: [distance, rate])
        dropout:           Dropout 概率 (默认 0.0)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        edge_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim)

        # ── 从观测空间推断图规模 ──
        M = observation_space["sat_features"].shape[0]
        I_max = observation_space["task_features"].shape[0]
        d_sat = observation_space["sat_features"].shape[1]    # 1
        d_task = observation_space["task_features"].shape[1]  # 4

        self.M = M
        self.I_max = I_max

        # ── 节点编码器: 原始特征 → 隐藏维度 ──
        self.sat_encoder = nn.Sequential(
            nn.Linear(d_sat, hidden_dim),
            nn.ELU(),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(d_task, hidden_dim),
            nn.ELU(),
        )

        # ── 异构 GAT 层 ──
        self.gnn_layers = nn.ModuleList([
            HeteroGATBlock(hidden_dim, num_heads, edge_dim, dropout)
            for _ in range(num_gnn_layers)
        ])

        # ── Readout MLP ──
        # 输入: 所有 task embeddings + 全局 sat pooling + num_tasks
        readout_in_dim = hidden_dim * I_max + hidden_dim + 1
        self.output_mlp = nn.Sequential(
            nn.Linear(readout_in_dim, 256),
            nn.ELU(),
            nn.Linear(256, features_dim),
            nn.ELU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """
        前向传播: Dict 观测 → 固定维度特征向量。

        Args:
            observations: Dict[str, Tensor]，每个值的 shape 为 (B, ...)

        Returns:
            features: (B, features_dim)
        """
        # ── 从 Dict 中提取各组件 ──
        sat_feat      = observations["sat_features"]       # (B, M+1, 1)
        task_feat     = observations["task_features"]      # (B, I_max, 4)
        task_sat_adj  = observations["task_sat_adj"]       # (B, I_max, M+1)
        task_sat_dist = observations["task_sat_dist"]      # (B, I_max, M+1)
        task_sat_rate = observations["task_sat_rate"]      # (B, I_max, M+1)
        sat_sat_adj   = observations["sat_sat_adj"]        # (B, M+1, M+1)
        sat_sat_dist  = observations["sat_sat_dist"]       # (B, M+1, M+1)
        sat_sat_rate  = observations["sat_sat_rate"]       # (B, M+1, M+1)
        num_tasks     = observations["num_tasks"]          # (B, 1)

        B = sat_feat.shape[0]

        # ── 1. 构造边特征张量 ──
        task_sat_edge_feat = torch.stack(
            [task_sat_dist, task_sat_rate], dim=-1
        )  # (B, I_max, M, 2)
        sat_sat_edge_feat = torch.stack(
            [sat_sat_dist, sat_sat_rate], dim=-1
        )  # (B, M, M, 2)

        # ── 2. 节点特征编码 ──
        task_h = self.task_encoder(task_feat)  # (B, I_max, hidden_dim)
        sat_h = self.sat_encoder(sat_feat)     # (B, M, hidden_dim)

        # ── 3. 多层异构 GAT 消息传递 ──
        for gnn_layer in self.gnn_layers:
            task_h, sat_h = gnn_layer(
                task_h, sat_h,
                task_sat_adj, task_sat_edge_feat,
                sat_sat_adj, sat_sat_edge_feat,
            )

        # ── 4. Readout ──
        task_flat = task_h.reshape(B, -1)     # (B, I_max * hidden_dim)
        sat_pooled = sat_h.mean(dim=1)        # (B, hidden_dim)

        combined = torch.cat([task_flat, sat_pooled, num_tasks], dim=-1)
        return self.output_mlp(combined)
