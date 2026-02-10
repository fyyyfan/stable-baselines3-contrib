import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.stats import poisson

try:
    from .coreForSat import (
        Time,
        Walker,
        SatelliteWorld,
        IoTDevice,
        SatMECNode,
        Task,
        CloudServer,
    )
except ImportError:
    from coreForSat import (
        Time,
        Walker,
        SatelliteWorld,
        IoTDevice,
        SatMECNode,
        Task,
        CloudServer,
    )


class SatelliteSingleAgentEnv(gym.Env):
    """
    用于训练的单智能体卫星网络任务卸载环境（适配 MaskablePPO 算法）。
    智能体作为全局任务调度器，在每个时隙对所有新生成的任务统一做出卸载决策。

    动作空间设计（MultiDiscrete）：
        M 颗卫星 → 每个任务有 B = M+2 种选择:
            0       : 本地执行
            1..M    : 卸载到卫星 1..M
            M+1     : 卸载到云端
        
        最大并发任务数 I_max，动作空间 = MultiDiscrete([B] * I_max)
        action[i] 表示第 i 个任务的卸载目标。
    
    动作掩码（action_masks）：
        返回形状为 (I_max * B,) 的布尔数组，用于屏蔽不满足约束的动作选项。
        约束包括：
        - 卫星可见性约束：设备必须在目标卫星的可见范围内
        - 卫星容量约束：卫星缓冲区不能溢出
    """
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None,
                 num_satellites: int = 5,
                 num_users: int = 10,
                 lambda0: float = 0.5,
                 I_max: int | None = None,
                 poisson_quantile: float = 0.95,
                 max_steps: int = 100,
                 env_update_interval: int = 5, #环境更新间隔，单位为时隙, dt*interval = 卫星网络更新间隔(s)
                 verbose: bool = True,  # 是否打印详细日志，训练时建议关闭
                 ):
        super().__init__()

        # ===================== 环境规模参数 =====================
        self.num_satellites = num_satellites   # M: 卫星数
        self.num_users = num_users             # N: IoT 设备数
        self.lambda0 = lambda0                 # 单个设备任务到达率
        self.max_steps = max_steps
        self.env_update_interval = env_update_interval
        self.verbose = verbose

        # ===================== 动作空间设计 =====================
        # 每个任务的卸载选择数: 本地(0) + M颗卫星(1..M) + 云端(M+1)
        self.B = self.num_satellites + 2       # 每个子动作的选项数 B = M+2

        # 计算 I_max: 最大并发任务数
        # μ = N * λ0，取泊松分布的 poisson_quantile 分位数
        self.mu = self.num_users * self.lambda0  # 所有用户单位时间内平均总任务数
        if I_max is not None:
            self.I_max = I_max
        else:
            # 自动计算: P(X ≤ I_max) ≥ poisson_quantile
            self.I_max = int(poisson.ppf(poisson_quantile, self.mu))
            self.I_max = max(self.I_max, 1)  # 至少 1

        # 多维离散动作空间: 每个任务一个子动作，每个子动作有 B 个选项
        # action = [a_0, a_1, ..., a_{I_max-1}], 其中 a_i ∈ {0, 1, ..., B-1}
        self.action_space = spaces.MultiDiscrete([self.B] * self.I_max)

        if self.verbose:
            print(f"[Env Init] M={self.num_satellites}, N={self.num_users}, "
                  f"λ0={self.lambda0}, μ={self.mu}, I_max={self.I_max}, "
                  f"B={self.B}, action_space=MultiDiscrete([{self.B}] * {self.I_max})")

        # ===================== 观测空间设计 =====================
        # 每个待决策任务: [task_size, cycles, deadline]    → I_max * 3 维
        # 候选卸载节点的状态：距离、传输速率、计算资源  I_max * B * 3 维
        self.obs_task_dim = 3
        self.obs_candi_state_dim = 3
        obs_size = (self.I_max * self.obs_task_dim + 
                    self.I_max * self.B * self.obs_candi_state_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        # ===================== 奖励权重 =====================
        # 第一组10,1.0
        self.delay_weight = 1.0
        self.energy_weight = 0.3
        self.overflow_penalty = 1.0

        # ===================== 运行时状态 =====================
        self.current_step = 0
        self.render_mode = render_mode
        self.world: SatelliteWorld | None = None
        self.cloud: CloudServer | None = None

        # 当前时隙的全局任务池（在 step 中填充）
        self.current_task_pool: list[tuple[IoTDevice, Task]] = []

    # ======================== 编解码 ========================

    def encode_action(self, decisions: list[int]) -> np.ndarray:
        """
        将卸载决策列表编码为 MultiDiscrete 动作数组。
        对于 MultiDiscrete 动作空间，动作本身就是数组形式，直接转换即可。
        
        Args:
            decisions: 长度为 I_max 的列表，decisions[i] ∈ {0, 1, ..., B-1}
        
        Returns:
            np.ndarray: 形状为 (I_max,) 的动作数组
        """
        assert len(decisions) == self.I_max
        return np.array(decisions, dtype=np.int64)

    def decode_action(self, action: np.ndarray) -> list[int]:
        """
        将 MultiDiscrete 动作数组解码为卸载决策列表。
        对于 MultiDiscrete 动作空间，动作本身就是数组形式，直接转换即可。
        
        Args:
            action: 形状为 (I_max,) 的动作数组
        
        Returns:
            decisions: 长度为 I_max 的列表，decisions[i] = 第 i 个任务的卸载目标
        """
        if isinstance(action, np.ndarray):
            return action.tolist()
        elif isinstance(action, (list, tuple)):
            return list(action)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

    # ======================== 观测 ========================

    def _get_obs(self) -> np.ndarray:
        """
        构造观测向量 S_t：
        [sat_0_queue, ..., sat_{M-1}_queue,
         num_tasks,
         task_0_size, task_0_cycles, task_0_deadline,
         ...,
         task_{I_max-1}_size, task_{I_max-1}_cycles, task_{I_max-1}_deadline]
        不足 I_max 的任务位置用 0 填充。
        
        【优化】预先缓存设备最近卫星，避免重复查询
        """
        if self.world is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs_parts = []

        # 1. 每个任务的特征，不足 I_max 补零
        task_features = np.zeros(self.I_max * self.obs_task_dim, dtype=np.float32)
        for idx, (_, task) in enumerate(self.current_task_pool[:self.I_max]):
            base = idx * self.obs_task_dim
            task_features[base] = float(task.task_size) / 1e2 #归一化
            task_features[base + 1] = float(task.total_cpu_cycles()) * 10 #归一化
            task_features[base + 2] = float(task.delay_requirement)
        obs_parts.append(task_features)

        # 【优化】预先计算所有设备的最近卫星，避免在循环中重复计算
        device_best_sat_cache = {}
        for device, _ in self.current_task_pool[:self.I_max]:
            if device.id not in device_best_sat_cache:
                device_best_sat_cache[device.id] = self.world._update_device_current_sat(device)
        
        # 【优化】缓存云端服务器相关信息（只计算一次）
        cloud_current_sat = self.world.cloud_server.current_sat if self.world.cloud_server else None
        cloud_f = self.world.cloud_server.f_cloud if self.world.cloud_server else 0.0

        # 2. 候选节点状态，不足I_max补零
        candi_state_features = np.zeros(self.I_max * self.B * self.obs_candi_state_dim, dtype=np.float32)
        for idx, (device, task) in enumerate(self.current_task_pool[:self.I_max]):
            base = idx * self.B * self.obs_candi_state_dim
            for target in range(self.B):
                offset = base + target * self.obs_candi_state_dim
                
                # 本地执行
                if target == 0:
                    candi_state_features[offset] = 0.0
                    candi_state_features[offset + 1] = 0.0
                    candi_state_features[offset + 2] = device.f_local
                
                # 卸载到卫星节点
                elif 1 <= target <= self.num_satellites:
                    sat_idx = target - 1
                    sat = self.world.satellites[sat_idx]
                    visibility_dist = self.world.user_sat_visibility.get((device.id, sat_idx), -1)
                    
                    if visibility_dist > 0:
                        candi_state_features[offset] = visibility_dist / 1e3  # 归一化
                        candi_state_features[offset + 1] = self.world._transmission_rate(device, task, sat) / 1e6  # Mbps
                    else:
                        candi_state_features[offset] = 0.0
                        candi_state_features[offset + 1] = 0.0
                    candi_state_features[offset + 2] = sat.comp_resource
                
                # 卸载到云中心
                else:
                    best_sat = device_best_sat_cache.get(device.id)
                    if best_sat is None or cloud_current_sat is None:
                        candi_state_features[offset] = 0.0
                        candi_state_features[offset + 1] = 0.0
                    else:
                        candi_state_features[offset] = self.world.get_backhaul_distance(device, self.world.cloud_server, best_sat) / 1e3
                        candi_state_features[offset + 1] = self.world._transmission_rate(self.world.cloud_server, task, cloud_current_sat) / 1e6
                    candi_state_features[offset + 2] = cloud_f
        
        obs_parts.append(candi_state_features)
        
        return np.concatenate(obs_parts).astype(np.float32)

    # ======================== 动作掩码 ========================
    def action_masks(self) -> np.ndarray:
        """
        生成动作掩码，用于 MaskablePPO 算法屏蔽不满足约束的动作。
        
        对于 MultiDiscrete([B] * I_max) 动作空间，返回扁平化的掩码数组，
        形状为 (I_max * B,)，布局为：
        [task_0 的 B 个选项掩码, task_1 的 B 个选项掩码, ..., task_{I_max-1} 的 B 个选项掩码]
        
        约束条件：
        1. 每个任务只能有一个卸载目标（由 MultiDiscrete 动作空间自然满足）
        2. 如果任务 i 卸载到卫星节点，产生任务 i 的 IoT 设备必须在卫星的 visible_user 列表中
        3. SatMECNode 节点当前处理的任务大小（queue_backlog）不能超过节点的存储资源（buffer_capacity）
        
        Returns:
            mask: shape (I_max * B,), True 表示有效动作，False 表示无效动作
        """
        # 初始化掩码矩阵: (I_max, B)，每行对应一个任务的所有选项
        mask_matrix = np.ones((self.I_max, self.B), dtype=bool)
        
        if self.world is None or len(self.current_task_pool) == 0:
            # 如果还没有初始化或没有任务，所有动作都视为有效
            return mask_matrix.flatten()
        
        # 获取当前任务数量
        num_tasks = min(len(self.current_task_pool), self.I_max)
        
        # 为每个真实任务计算掩码
        for i in range(num_tasks):
            device, task = self.current_task_pool[i]
            task_cycles = task.total_cpu_cycles()
            
            # 检查每个可能的卸载目标
            for target in range(self.B):
                valid = True
                
                # target = 0: 本地执行（总是有效，假设本地设备有足够容量）
                if target == 0:
                    # 本地执行：检查设备本地队列+新任务是否会超过限制
                    # 注意：IoTDevice没有显式的存储限制，这里假设本地总是可以执行
                    # 如果需要限制本地存储，可以在这里添加检查
                    valid = True
                
                # target = 1..M: 卸载到卫星
                elif 1 <= target <= self.num_satellites:
                    sat_idx = target - 1
                    sat = self.world.satellites[sat_idx]
                    
                    if not isinstance(sat, SatMECNode):
                        valid = False
                    else:
                        # 约束2: 检查设备是否在卫星可见范围内
                        if device not in sat.visible_user:
                            valid = False
                        
                        
                        # 约束3: 检查卫星存储资源是否足够
                        # 当前队列积压 + 新任务计算量 不能超过缓冲区容量
                        if valid and sat.queue_backlog + task_cycles > sat.buffer_capacity:
                            valid = False
                
                # target = M+1: 卸载到云端
                elif target == self.num_satellites + 1:
                    # 如果用户没有可见卫星
                    if self.world._update_device_current_sat(device) is None:
                        valid = False
                    else:
                        valid = True
                
                mask_matrix[i, target] = valid
        
        # 对于超出实际任务数的虚拟任务位置，所有目标都标记为有效
        
        # 扁平化返回: (I_max * B,)
        return mask_matrix.flatten()
    
    def _get_per_task_masks(self) -> np.ndarray:
        """
        返回二维掩码矩阵，方便调试和分析。
        
        Returns:
            mask_matrix: shape (I_max, B)，mask_matrix[i, j] 表示任务 i 能否选择目标 j
        """
        flat_mask = self.action_masks()
        return flat_mask.reshape(self.I_max, self.B)
    
    
    def _get_info(self) -> dict:
        return {"current_step": self.current_step}

    # ======================== reset ========================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_task_pool = []

        # 1. 创建物理世界时间
        start_time = Time(year=2024, month=1, day=3, hour=8, minute=0, second=0.0)

        # 2. 创建 Walker 星座与卫星
        walker = Walker(
            num_sats=self.num_satellites,
            h=7000,
            angle=51.664,
            P_num=1,
            sat_tran_power=np.random.randint(5, 15, self.num_satellites),#[10] * self.num_satellites,
            sat_tran_gain=np.random.randint(30, 45, self.num_satellites),#[36] * self.num_satellites,
            sat_rec_gain=np.random.randint(36, 51, self.num_satellites),#[41] * self.num_satellites,
            sat_comp_resource=np.random.randint(2, 8, self.num_satellites),#[5]*self.num_satellites,
            sat_kappa_sat=[1e-28]*self.num_satellites,
            sat_buffer_capacity=[10] * self.num_satellites,#.random.randint(200, 400, self.num_satellites),
        )
        satellites = walker.create_satellites()

        # 3. 创建世界对象并挂载卫星
        self.world = SatelliteWorld(walker, start_time)
        self.world.satellites = satellites

        # 4. 创建 IoT 设备
        self.world.user_clusters = []
        
        for uid in range(self.num_users):
            
            lon = [100, 102, 105, 110, 115, 120, 123, 128, 130, 131]#int(np.random.uniform(110, 125))
            lat = [40, 38, 45, 35, 32, 47, 42, 36, 50, 48]#int(np.random.uniform(42,48))
            device = IoTDevice(
                id=uid,
                lon=lon[uid],
                lat=lat[uid],
                lambda0=self.lambda0,
                f_local=2,
                p_tx=0.5,
                kappa_ue=5e-28
            )
            self.world.user_clusters.append(device)

        # 5. 创建云端服务器
        self.world.cloud_server = CloudServer(
            id=self.num_satellites + 1,
            lon=120,
            lat=45,
            f_cloud=10
        )

        # 6. 初始拓扑/可见性更新
        self.world._update_link_states(self.world.current_time)
        self.world._update_visibility_matrix(self.world.current_time)
        # 更新云端服务器当前卫星
        self.world._update_cloud_current_sat(self.world.cloud_server)

        # 7. 生成第一批任务，填充任务池，以便 agent 拿到初始观测
        self._generate_and_collect_tasks()

        # ==============记录观测============================
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    # ======================== step ========================

    def step(self, action):
        self.current_step += 1
        assert self.world is not None, "Call reset() first."
        w = self.world
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Step {self.current_step}] 开始执行")
            print(f"{'='*60}")

        # ========== 1. 更新世界时间与轨迹 ==========
        w.world_step += 1
        w.current_time.second += w.dt
        if w.current_time.second >= 60:
            w.current_time.minute += int(w.current_time.second // 60)
            w.current_time.second = w.current_time.second % 60
        if w.current_time.minute >= 60:
            w.current_time.hour += int(w.current_time.minute // 60)
            w.current_time.minute = w.current_time.minute % 60
        if w.current_time.hour >= 24:
            w.current_time.day += int(w.current_time.hour // 24)
            w.current_time.hour = w.current_time.hour % 24

        if self.verbose:
            print(f"[时间更新] {w.current_time.hour}:{w.current_time.minute}:{w.current_time.second:.1f}")
            print(f"[当前任务池] 任务数量: {len(self.current_task_pool)}")
            for idx, (dev, task) in enumerate(self.current_task_pool[:min(len(self.current_task_pool), self.I_max)]):
                print(f"  任务{idx}: 设备{dev.id}, size={task.task_size:.2f}kbits, "
                      f"cycles={task.total_cpu_cycles():.2f}Gcycles")

        # ========== 2. 解码动作 ==========
        # action 是 MultiDiscrete 动作，形状为 (I_max,)
        # action[i] ∈ {0..B-1} 对应 current_task_pool[i] 的卸载目标
        decisions = self.decode_action(action)
        if self.verbose:
            print(f"\n[动作解码] decisions = {decisions[:min(len(self.current_task_pool), self.I_max)]}")

        # ========== 3. 对当前任务池中的每个任务执行卸载决策, 更新节点资源状态==
        total_delay = 0.0
        total_energy = 0.0
        overflow_count = 0
        num_tasks_decided = min(len(self.current_task_pool), self.I_max)

        for i in range(num_tasks_decided):
            device, task = self.current_task_pool[i]
            target = decisions[i]  # 0=本地, 1..M=卫星, M+1=云

            delay_i, energy_i, overflow_i = self._execute_offload(
                device, task, target, w
            )
            if self.verbose:
                print(f"【任务{i}】延迟: {delay_i:.2f}s, 能耗: {energy_i:.2f}J, 是否溢出: {overflow_i}")
            # TODO:奖励计算待修改
            total_delay += delay_i
            total_energy += energy_i
            if overflow_i:
                overflow_count += 1

        # ===========消耗卫星节点队列中的任务，通过消耗 w.dt 计算量，任务完成后更新 service_users）
        for sat in w.satellites[: self.num_satellites]:
            if isinstance(sat, SatMECNode):
                old_backlog = sat.queue_backlog
                old_service_users = [u.id for u in sat.service_users]
                _, processed = sat.process_queue(w.dt)
                new_service_users = [u.id for u in sat.service_users]
                # if processed > 0 or old_service_users != new_service_users:
                #     print(f"  卫星{sat.id}: 处理了{processed:.4f}cycles, "
                #           f"queue_backlog: {old_backlog:.4f}->{sat.queue_backlog:.4f}, "
                #           f"service_users: {old_service_users}->{new_service_users}")

        # ========== 4. 计算奖励 ==========
        if num_tasks_decided == 0:
            reward = 0.0
        else:
            reward = -(self.delay_weight * total_delay
                   + self.energy_weight * total_energy
                   + self.overflow_penalty * overflow_count) / num_tasks_decided
        if self.verbose:
            print(f"\n[奖励计算] total_delay={total_delay:.2f}s, total_energy={total_energy:.2f}J, "
                  f"overflow_count={overflow_count}, reward={reward:.2f}")

        # ========== 5. 为下一时隙生成新任务并收集任务池 ==========
        self._generate_and_collect_tasks()

        # 一定间隔更新卫星网络拓扑、可见性、云端服务器当前卫星，不会每步更新
        if self.current_step % self.env_update_interval == 0:
            # 更新 ISL 拓扑和用户-卫星可见性
            w._update_link_states(w.current_time)
            w._update_visibility_matrix(w.current_time)
            # 更新云端服务器当前卫星
            w._update_cloud_current_sat(self.world.cloud_server)
            # print(f"=============[环境更新] 更新卫星网络拓扑、可见性、云端服务器当前卫星=============")


        # ========== 6. 终止条件 ==========
        terminated = self.current_step >= self.max_steps
        truncated = False

        observation = self._get_obs()
        info = self._get_info()
        info.update({
            "total_delay": float(total_delay),
            "total_energy": float(total_energy),
            "overflow_count": int(overflow_count),
            "num_tasks": num_tasks_decided,
        })

        return observation, reward, terminated, truncated, info

    # ======================== 内部辅助方法 ========================

    def _generate_and_collect_tasks(self):
        """
        让所有 IoT 设备生成新任务，然后从各设备队列中收集待决策任务，
        汇总到 self.current_task_pool（最多 I_max 个）。
        """
        assert self.world is not None
        # 所有设备产生新任务
        for device in self.world.user_clusters:
            device.generate_tasks(tau=self.world.dt)

        # 从各设备队列头部收集，总共最多 I_max 个
        self.current_task_pool = []
        cnt = 0 #本次生成任务数量
        for device in self.world.user_clusters:
            while device.task_queue and len(self.current_task_pool) < self.I_max:
                task = device.task_queue.pop(0)
                self.current_task_pool.append((device, task))
                cnt += 1
            if len(self.current_task_pool) >= self.I_max:
                break
        if self.verbose:
            print(f"[任务生成] 收集了 {cnt} 个新任务到任务池 (I_max={self.I_max})")

    def _execute_offload(self, device: IoTDevice, task: Task,
                         target: int, w: SatelliteWorld
                         ) -> tuple[float, float, bool]:
        """
        执行单个任务的卸载决策，并更新设备/卫星节点资源占用。
        若目标节点资源不足，不执行卸载，返回 overflow=True 并计入惩罚。

        Args:
            device: 产生该任务的 IoT 设备
            task: 待卸载的任务
            target: 卸载目标
                0       → 本地执行
                1..M    → 卸载到卫星 (target-1)
                M+1     → 卸载到云端
            w: SatelliteWorld 引用

        Returns:
            (delay, energy, overflow_flag)
        """
        delay = 0.0
        energy = 0.0
        overflow = False

        M = self.num_satellites
        need_cycles = task.total_cpu_cycles()

        # ---------- 本地执行 ----------
        if target == 0:
            delay = w.compute_local_delay(device, task)
            energy = w.compute_local_energy(device, task)

        # ---------- 卸载到卫星 ----------
        elif 1 <= target <= M:
            sat_idx = target - 1
            sat = w.satellites[sat_idx]
            assert isinstance(sat, SatMECNode)

            # 1) 绑定当前服务卫星，并先加入 service_users（仅用于本步延迟/能耗计算中的 f_comp 分摊）
            device.current_sat = sat
            device_was_in_service = device in sat.service_users
            if not device_was_in_service:
                sat.service_users.append(device)
                # print(f"[service_users 更新] 添加设备{device.id}到卫星{sat.id}的service_users")

            # 2) 先计算延迟与能耗（此时 get_queue_delay() 看到的是本任务入队前的积压，排队时延不含本任务）
            delay = w.compute_edge_delay(device, task, sat)
            energy = w.compute_edge_energy(device, task, sat)

            # 3) 再入队：整包拒绝时从 service_users 移除（仅当该设备在队列中没有其他任务时）
            arrival_cycles = task.total_cpu_cycles()
            overflow = sat.enqueue_task(device, arrival_cycles)
            
            if overflow and self.verbose:
                print(f" [入队失败] 设备{device.id}任务入队卫星{sat.id}失败(缓冲区溢出)")
                # 修复 Bug: 只有当该设备在队列中没有其他任务时，才从 service_users 移除
                has_other_tasks_in_queue = any(d is device for d, _ in sat.task_queue)
                if device in sat.service_users and not has_other_tasks_in_queue:
                    sat.service_users.remove(device)
                    # print(f" [service_users 更新] 从卫星{sat.id}的service_users中移除设备{device.id}")
                # elif has_other_tasks_in_queue:
                    # print(f" [service_users 保留] 设备{device.id}在卫星{sat.id}队列中仍有其他任务，保留在service_users中")
            else:
                pass
                # print(f"[入队成功] 设备{device.id}任务入队卫星{sat.id}成功, 当前queue_backlog={sat.queue_backlog:.4f}")
            
        # ---------- 卸载到云端 ----------
        elif target == M + 1:
            # 给用户设置当前最近卫星
            device.current_sat = w._update_device_current_sat(device)
            if w.cloud_server is not None:
                if w.cloud_server.current_sat is None and self.verbose:
                    print("云端当前卫星为None")
                delay = w.compute_cloud_total_delay(device, task, w.cloud_server)
                energy = w.compute_cloud_energy(device, task, w.cloud_server)
            # else:
            #     print("云端未创建")
        
        # 检查延迟是否超过任务截止时间
        if delay > task.delay_requirement:
                overflow = True

        return delay, energy, overflow

    # ======================== 渲染 / 关闭 ========================

    def render(self):
        if self.render_mode == "human":
            pool_info = [(d.id, t.task_size) for d, t in self.current_task_pool]
            print(f"[Step {self.current_step}] task_pool({len(self.current_task_pool)}): {pool_info}")
        elif self.render_mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self):
        pass
