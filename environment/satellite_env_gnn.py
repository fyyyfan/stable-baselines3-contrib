"""
GNN 版本的卫星任务卸载环境 —— 异构二部图观测空间
=================================================

继承 SatelliteSingleAgentEnv，仅修改观测空间和观测构建方式，
其余逻辑（动作空间、掩码、step、奖励等）完全复用基类实现。

观测空间变更:
  - 基类: spaces.Box (扁平化向量)
  - 本类: spaces.Dict (异构二部图结构信息)

异构二部图定义:
  节点类型:
    - "sat"  节点 (M 个):  特征 = [queue_backlog, comp_resource, buffer_capacity, buffer_usage_ratio]
    - "task" 节点 (I_max 个): 特征 = [task_size, cycles, f_local, is_real_task]
                               (虚拟任务位 is_real_task=0, 真实任务=1)
  边类型:
    - "task→sat" (用户-卫星上行链路):
        条件: task_i 的设备对 sat_j 可见
        边特征 = [normalized_distance, estimated_transmission_rate]
    - "sat→sat" (星间链路 ISL):
        来源: self.world.sat_topology / self.world.sat_links
        边特征 = [normalized_distance, data_rate]

设计原则:
  1. 不影响原有 SatelliteSingleAgentEnv 的任何行为
  2. 动作空间、动作掩码、step 逻辑完全继承
  3. 仅覆写 __init__ (修改 observation_space) 和 _get_obs (构建 Dict 观测)
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

try:
    from .satellite_single_agent_env import SatelliteSingleAgentEnv
    from .coreForSat import SatMECNode
except ImportError:
    from satellite_single_agent_env import SatelliteSingleAgentEnv
    from coreForSat import SatMECNode


class SatelliteGNNEnv(SatelliteSingleAgentEnv):
    """
    GNN 版本的卫星任务卸载环境。

    与基类 SatelliteSingleAgentEnv 的唯一区别:
      - 观测空间由 Box(flat_vector) 变为 Dict(graph_components)
      - _get_obs() 返回异构二部图的节点特征、邻接矩阵、边特征

    动作空间、动作掩码、step 逻辑、奖励计算完全复用基类。

    Args:
        与 SatelliteSingleAgentEnv 完全相同
    """

    def __init__(self, **kwargs):
        # 调用基类 __init__ (会设置 observation_space 为 Box)
        super().__init__(**kwargs)

        M = self.num_satellites
        I_max = self.I_max
        sat_node_count = M + int(self.enable_cloud_offload)

        # ── 覆盖观测空间为 Dict ──
        self.observation_space = spaces.Dict({
            # 卫星节点特征: [comp_resource, remaining_capacity_ratio]
            "sat_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(sat_node_count, 2), dtype=np.float32
            ),
            
            # 任务节点特征: [task_size, cycles, f_local, is_real_task]
            "task_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(I_max, 4), dtype=np.float32
            ),
            # 任务→卫星邻接矩阵 (二值: 1=可见, 0=不可见)
            "task_sat_adj": spaces.Box(
                low=0.0, high=1.0,
                shape=(I_max, sat_node_count), dtype=np.float32
            ),
            # 任务→卫星归一化距离
            "task_sat_dist": spaces.Box(
                low=0.0, high=np.inf,
                shape=(I_max, sat_node_count), dtype=np.float32
            ),
            # 任务→卫星估计上行传输速率
            "task_sat_rate": spaces.Box(
                low=0.0, high=np.inf,
                shape=(I_max, sat_node_count), dtype=np.float32
            ),
            # 卫星间 ISL 邻接矩阵 (二值)
            "sat_sat_adj": spaces.Box(
                low=0.0, high=1.0,
                shape=(sat_node_count, sat_node_count), dtype=np.float32
            ),
            # 卫星间归一化距离
            "sat_sat_dist": spaces.Box(
                low=0.0, high=np.inf,
                shape=(sat_node_count, sat_node_count), dtype=np.float32
            ),
            # 卫星间 ISL 数据传输速率
            "sat_sat_rate": spaces.Box(
                low=0.0, high=np.inf,
                shape=(sat_node_count, sat_node_count), dtype=np.float32
            ),
            # 实际任务数 (归一化到 [0, 1])
            "num_tasks": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,), dtype=np.float32
            ),
        })

        if self.verbose:
            print(f"[GNN Env Init] 观测空间已变更为 Dict:")
            for key, space in self.observation_space.spaces.items():
                print(f"  {key}: {space.shape}")

    # ======================== 覆写 _get_obs ========================

    def _get_obs(self) -> dict:
        """
        构造异构二部图观测 (Dict)。

        Returns:
            dict: 包含以下键的字典
              - sat_features:  (M 或 M+1, 2) 卫星节点特征
              - task_features: (I_max, 4)    任务节点特征 [task_size, cycles, f_local, is_real_task]
              - task_sat_adj:  (I_max, M 或 M+1)  任务→卫星邻接矩阵
              - task_sat_dist: (I_max, M 或 M+1)  归一化星地距离
              - task_sat_rate: (I_max, M 或 M+1)  上行传输速率 (Mbps)
              - sat_sat_adj:   (M 或 M+1, M 或 M+1) 卫星间 ISL 邻接矩阵
              - sat_sat_dist:  (M 或 M+1, M 或 M+1) 归一化星间距离
              - sat_sat_rate:  (M 或 M+1, M 或 M+1) ISL 速率 (Gbps)
              - num_tasks:     (1,)          实际任务数 / I_max
        """
        M = self.num_satellites
        I_max = self.I_max
        sat_node_count = M + int(self.enable_cloud_offload)
        cloud_node_idx = M if self.enable_cloud_offload else None

        # ── 默认全零观测 (world 未初始化时) ──
        if self.world is None:
            return {
                key: np.zeros(space.shape, dtype=np.float32)
                for key, space in self.observation_space.spaces.items()
            }

        w = self.world

        # ============================================================
        # 1. 卫星节点特征: [comp_resource, remaining_capacity_ratio]
        #    comp_resource:             处理速率 (GCycles/s)，归一化
        #    remaining_capacity_ratio:  (buffer_capacity - queue_backlog) / buffer_capacity
        #                               1.0 = 空闲，0.0 = 满载
        # ============================================================
        sat_features = np.zeros((sat_node_count, 2), dtype=np.float32)
        for j in range(M):
            sat = w.satellites[j]
            if isinstance(sat, SatMECNode):
                sat_features[j, 0] = sat.comp_resource / 10
                if len(sat.visible_user) > 0:
                    sat_features[j, 1] = 0.0 #len(sat.visible_user) / 10 #可见用户数量
                else:
                    sat_features[j, 1] = 0.0
        if self.enable_cloud_offload and w.cloud_server is not None and cloud_node_idx is not None:
            sat_features[cloud_node_idx, 0] = w.cloud_server.f_cloud / 10
            sat_features[cloud_node_idx, 1] = 1.0  # 云端视为无限容量，剩余比率恒为 1
        # ============================================================
        # 2. 任务节点特征: [task_size, cycles, f_local, is_real_task]
        # ============================================================
        task_features = np.zeros((I_max, 4), dtype=np.float32)
        num_real_tasks = min(len(self.current_task_pool), I_max)

        for i in range(num_real_tasks):
            device, task = self.current_task_pool[i]
            task_features[i, 0] = float(task.task_size) / 1e3         # kbits → 归一化
            task_features[i, 1] = float(task.total_cpu_cycles())      # Gcycles → 归一化
            task_features[i, 2] = float(device.f_local) / 10.0        # GHz → 归一化
            task_features[i, 3] = 1.0   # is_real_task
        # 超出 num_real_tasks 的槽位保持 0 (is_real_task=0, f_local=0)

        # ============================================================
        # 3. task→sat 边 (用户-卫星上行链路可见性)
        #    条件: task_i 的设备对 sat_j 可见
        #    边特征: [normalized_distance, estimated_transmission_rate]
        # ============================================================
        task_sat_adj  = np.zeros((I_max, sat_node_count), dtype=np.float32)
        task_sat_dist = np.zeros((I_max, sat_node_count), dtype=np.float32)
        task_sat_rate = np.zeros((I_max, sat_node_count), dtype=np.float32)

        # 【优化】预先计算所有设备的最近卫星，避免在循环中重复计算
        device_best_sat_cache = {}
        for device, _ in self.current_task_pool[:self.I_max]:
            if device.id not in device_best_sat_cache:
                device_best_sat_cache[device.id] = self.world._update_device_current_sat(device)
        # 【优化】缓存云端服务器相关信息（只计算一次）
        cloud_current_sat = (
            self.world.cloud_server.current_sat
            if self.enable_cloud_offload and self.world.cloud_server is not None
            else None
        )

        #计算边特征
        for i in range(num_real_tasks):
            device, task = self.current_task_pool[i]
            for j in range(M):
                dist = w.user_sat_visibility.get((device.id, j), -1)
                if dist > 0:
                    task_sat_adj[i, j] = 1.0
                    dist_norm = float((dist - 500) / (2000 - 500))  # km → 归一化dist/1e4
                    rate = w._transmission_rate(device, task, w.satellites[j])
                    rate_norm = float(rate / 1e7)                # bit/s → Mbps
                    if np.isfinite(dist_norm):
                        task_sat_dist[i, j] = dist_norm
                    if np.isfinite(rate_norm):
                        task_sat_rate[i, j] = rate_norm
            # 单独处理云端节点（可选）
            if self.enable_cloud_offload and cloud_node_idx is not None:
                best_sat = device_best_sat_cache.get(device.id)
                if best_sat is None or cloud_current_sat is None:
                    task_sat_adj[i, cloud_node_idx] = 0.0
                    task_sat_dist[i, cloud_node_idx] = 0.0
                    task_sat_rate[i, cloud_node_idx] = 0.0
                else:
                    cloud_dist = float(
                        self.world.get_backhaul_distance(device, self.world.cloud_server, best_sat) / 1e4
                    )
                    cloud_rate = float(
                        self.world._transmission_rate(self.world.cloud_server, task, cloud_current_sat) / 1e7
                    )
                    # 云端路径不可达时禁止该动作，并保持观测为有限值，避免 NaN/Inf 污染网络。
                    if np.isfinite(cloud_dist) and np.isfinite(cloud_rate) and cloud_rate > 0.0:
                        task_sat_adj[i, cloud_node_idx] = 1.0
                        task_sat_dist[i, cloud_node_idx] = cloud_dist
                        task_sat_rate[i, cloud_node_idx] = cloud_rate
                    else:
                        task_sat_adj[i, cloud_node_idx] = 0.0
                        task_sat_dist[i, cloud_node_idx] = 0.0
                        task_sat_rate[i, cloud_node_idx] = 0.0

        # ============================================================
        # 4. sat→sat 边 (星间链路 ISL)
        #    来源: self.world.sat_links
        #    边特征: [normalized_distance, data_rate]
        # ============================================================
        sat_sat_adj  = np.zeros((sat_node_count, sat_node_count), dtype=np.float32)
        sat_sat_dist = np.zeros((sat_node_count, sat_node_count), dtype=np.float32)
        sat_sat_rate = np.zeros((sat_node_count, sat_node_count), dtype=np.float32)

        for (s1, s2), link_info in w.sat_links.items():
            if 0 <= s1 < M and 0 <= s2 < M:
                sat_sat_adj[s1, s2]  = 1.0
                sat_sat_dist[s1, s2] = link_info["distance"] / 1e4   # km → 归一化
                sat_sat_rate[s1, s2] = link_info["data_rate"] / 1e9   # bit/s → Gbps

        # ============================================================
        # 5. 实际任务数 (归一化)
        # ============================================================
        num_tasks = np.array(
            [num_real_tasks / max(I_max, 1)], dtype=np.float32
        )

        return {
            "sat_features":  sat_features,
            "task_features": task_features,
            "task_sat_adj":  task_sat_adj,
            "task_sat_dist": task_sat_dist,
            "task_sat_rate": task_sat_rate,
            "sat_sat_adj":   sat_sat_adj,
            "sat_sat_dist":  sat_sat_dist,
            "sat_sat_rate":  sat_sat_rate,
            "num_tasks":     num_tasks,
        }
