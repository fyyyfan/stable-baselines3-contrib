"""
SatelliteSingleAgentEnv 环境测试脚本
用于验证环境创建、reset 和 step 是否正常，为后续训练 PPO 算法做准备。

测试内容：
1. 环境创建和初始化
2. reset 方法测试
3. step 方法测试（使用满足动作掩码的随机策略）
4. 打印详细信息：任务状态、卫星状态、动作掩码、延迟能耗等
"""

import numpy as np
import gymnasium as gym
from satellite_single_agent_env import SatelliteSingleAgentEnv

# ======================== 工具函数 ========================

def print_separator(title: str, char: str = "=", length: int = 80):
    """打印分隔线"""
    print(f"\n{char * length}")
    print(f"  {title}")
    print(f"{char * length}")


def print_task_info(env: SatelliteSingleAgentEnv, verbose: bool = True):
    """打印当前任务池的详细信息"""
    task_pool = env.current_task_pool
    print(f"\n[任务池状态] 当前任务数: {len(task_pool)} / I_max={env.I_max}")
    
    if verbose and len(task_pool) > 0:
        print("-" * 60)
        print(f"{'任务ID':<8} {'设备ID':<8} {'大小(bits)':<15} {'CPU周期':<15} {'截止时间(s)':<12}")
        print("-" * 60)
        
        for idx, (device, task) in enumerate(task_pool[:env.I_max]):
            print(f"{idx:<8} {device.id:<8} {task.task_size:<15.2e} "
                  f"{task.computing_requirement:<15.2e} {task.delay_requirement:<12.4f}")
        print("-" * 60)


def print_satellite_info(env: SatelliteSingleAgentEnv, verbose: bool = True):
    """打印卫星状态的详细信息"""
    if env.world is None:
        print("[错误] 环境未初始化，world 为 None")
        return
    
    print(f"\n[卫星状态] 共 {env.num_satellites} 颗卫星")
    
    if verbose:
        print("-" * 80)
        print(f"{'卫星ID':<8} {'队列积压':<15} {'缓冲容量':<15} {'可见用户ID':<12} {'服务用户ID':<12}")
        print("-" * 80)
        
        for idx, sat in enumerate(env.world.satellites):
            queue_backlog = sat.queue_backlog
            buffer_cap = sat.buffer_capacity
            # # 确保为标量以便用 .2f 格式化（若为 list 则取和或首元素）
            # if isinstance(queue_backlog, list):
            #     queue_backlog = sum(queue_backlog) if queue_backlog else 0.0
            # else:
            #     queue_backlog = float(queue_backlog)
            # if isinstance(buffer_cap, list):
            #     buffer_cap = sum(buffer_cap) if buffer_cap else 0.0
            # else:
            #     buffer_cap = float(buffer_cap)
            visible_user_ids = [user.id for user in sat.visible_user] if hasattr(sat, 'visible_user') else []
            service_user_ids = [user.id for user in sat.service_users] if hasattr(sat, 'service_users') else []
            
            print(f"{idx:<8} {queue_backlog:<15.2f} {buffer_cap:<15.2f} "
                  f"{visible_user_ids!s:<12} {service_user_ids!s:<12}")
        print("-" * 80)


def print_action_masks(env: SatelliteSingleAgentEnv, verbose: bool = True):
    """打印动作掩码的详细信息"""
    mask_matrix = env._get_per_task_masks()  # shape: (I_max, B)
    num_tasks = min(len(env.current_task_pool), env.I_max)
    
    print(f"\n[动作掩码] 形状: ({env.I_max}, {env.B})")
    print(f"  动作含义: 0=本地, 1~{env.num_satellites}=卫星1~{env.num_satellites}, {env.B-1}=云端")
    
    if verbose and num_tasks > 0:
        print("-" * 80)
        header = f"{'任务ID':<8}"
        for j in range(env.B):
            if j == 0:
                header += f"{'本地':<8}"
            elif j <= env.num_satellites:
                header += f"{'卫星'+str(j):<8}"
            else:
                header += f"{'云端':<8}"
        print(header)
        print("-" * 80)
        
        for i in range(min(num_tasks, 10)):  # 最多显示10个任务
            row = f"{i:<8}"
            for j in range(env.B):
                status = "✓" if mask_matrix[i, j] else "✗"
                row += f"{status:<8}"
            
            # 显示该任务的可行动作数
            valid_count = mask_matrix[i].sum()
            row += f"  (可用: {valid_count}/{env.B})"
            print(row)
        
        if num_tasks > 10:
            print(f"  ... 省略 {num_tasks - 10} 个任务的掩码信息")
        print("-" * 80)
    
    return mask_matrix


def sample_masked_action(env: SatelliteSingleAgentEnv, mask_matrix: np.ndarray) -> np.ndarray:
    """
    根据动作掩码采样有效动作（随机策略）
    
    Args:
        env: 环境实例
        mask_matrix: 动作掩码矩阵，shape: (I_max, B)
    
    Returns:
        action: 满足掩码约束的动作数组，shape: (I_max,)
    """
    action = np.zeros(env.I_max, dtype=np.int64)
    
    for i in range(env.I_max):
        valid_actions = np.where(mask_matrix[i])[0]  # 获取有效动作索引
        if len(valid_actions) > 0:
            action[i] = np.random.choice(valid_actions)
        else:
            # 如果没有有效动作，默认选择本地执行
            action[i] = 0
    
    return action


def print_action_decision(env: SatelliteSingleAgentEnv, action: np.ndarray):
    """打印动作决策的详细信息"""
    num_tasks = min(len(env.current_task_pool), env.I_max)
    
    print(f"\n[卸载决策] 对 {num_tasks} 个任务进行决策")
    
    if num_tasks > 0:
        print("-" * 70)
        print(f"{'任务ID':<8} {'设备ID':<8} {'决策':<15} {'目标说明':<30}")
        print("-" * 70)
        
        decision_stats = {"本地": 0, "卫星": 0, "云端": 0}
        
        for i in range(num_tasks):
            device, task = env.current_task_pool[i]
            target = action[i]
            
            if target == 0:
                target_desc = "本地执行"
                decision_stats["本地"] += 1
            elif 1 <= target <= env.num_satellites:
                target_desc = f"卫星 {target}"
                decision_stats["卫星"] += 1
            else:
                target_desc = "云端"
                decision_stats["云端"] += 1
            
            print(f"{i:<8} {device.id:<8} {target:<15} {target_desc:<30}")
        
        print("-" * 70)
        print(f"[决策统计] 本地: {decision_stats['本地']}, "
              f"卫星: {decision_stats['卫星']}, 云端: {decision_stats['云端']}")


def print_observation(env: SatelliteSingleAgentEnv, obs: np.ndarray, verbose: bool = True):
    """打印观测向量的详细信息"""
    print(f"\n[观测向量] 形状: {obs.shape}")
    
    if verbose:
        # 解析观测向量结构
        sat_dim = env.num_satellites * env.obs_sat_dim
        num_tasks_idx = sat_dim
        task_start_idx = sat_dim + 1
        
        # 卫星负载
        sat_loads = obs[:sat_dim]
        print(f"  卫星队列负载: {sat_loads}")
        
        # 任务数量
        num_tasks = int(obs[num_tasks_idx])
        print(f"  当前任务数: {num_tasks}")
        
        # 任务特征（显示前几个）
        if num_tasks > 0:
            print(f"  任务特征 (size, cycles, deadline):")
            for i in range(min(num_tasks, 5)):
                base = task_start_idx + i * env.obs_task_dim
                task_size = obs[base]
                cycles = obs[base + 1]
                deadline = obs[base + 2]
                print(f"    任务{i}: [{task_size:.2e}, {cycles:.2f}, {deadline:.4f}]")
            if num_tasks > 5:
                print(f"    ... 省略 {num_tasks - 5} 个任务")


def print_step_result(step_num: int, obs: np.ndarray, reward: float, 
                     terminated: bool, truncated: bool, info: dict):
    """打印 step 执行结果的详细信息"""
    print(f"\n[Step {step_num} 结果]")
    print(f"  奖励: {reward:.6f}")
    print(f"  终止: {terminated}, 截断: {truncated}")
    print(f"  详细信息:")
    print(f"    - 总延迟: {info.get('total_delay', 0):.6f} s")
    print(f"    - 总能耗: {info.get('total_energy', 0):.6e} J")
    print(f"    - 溢出任务数: {info.get('overflow_count', 0)}")
    print(f"    - 处理任务数: {info.get('num_tasks', 0)}")


# ======================== 主测试函数 ========================

def test_env_creation():
    """测试1: 环境创建"""
    print_separator("测试1: 环境创建")
    
    print("\n[创建环境] 参数配置:")
    
    env = SatelliteSingleAgentEnv(
        num_satellites=4,
        num_users=10,
        lambda0=0.5,
        I_max=5,
        max_steps=20
    )
    
    print(f"\n[环境创建成功]")
    print(f"  动作空间: {env.action_space}")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作维度: I_max={env.I_max}, B={env.B}")
    
    return env


def test_env_reset(env: SatelliteSingleAgentEnv):
    """测试2: 环境重置"""
    print_separator("测试2: 环境重置 (reset)")
    
    print("\n[执行 reset()]...")
    obs, info = env.reset(seed=42)
    
    print(f"\n[Reset 成功]")
    print(f"  返回的 info: {info}")
    
    # 打印详细状态
    print_observation(env, obs)
    print_satellite_info(env)
    print_task_info(env)
    
    return obs, info


def test_env_step(env: SatelliteSingleAgentEnv, num_steps: int = 10):
    """测试3: 环境单步执行"""
    print_separator(f"测试3: 环境单步执行 (step) - 共 {num_steps} 步")
    
    total_reward = 0.0
    total_delay = 0.0
    total_energy = 0.0
    total_overflow = 0
    
    for step in range(1, num_steps + 1):
        print_separator(f"Step {step}/{num_steps}", char="-", length=60)
        
        # 1. 打印当前任务状态
        print_task_info(env)
        
        # 2. 打印卫星状态
        print_satellite_info(env)
        
        # 3. 获取并打印动作掩码
        mask_matrix = print_action_masks(env)
        
        # 4. 基于掩码采样动作（随机策略）
        action = sample_masked_action(env, mask_matrix)
        
        # 5. 打印卸载决策
        print_action_decision(env, action)
        
        # 6. 执行 step
        print(f"\n[执行 step]...")
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 7. 打印 step 结果
        print_step_result(step, obs, reward, terminated, truncated, info)
        
        # 8. 累积统计
        total_reward += reward
        total_delay += info.get('total_delay', 0)
        total_energy += info.get('total_energy', 0)
        total_overflow += info.get('overflow_count', 0)
        
        # 9. 打印新的观测
        print_observation(env, obs, verbose=(step <= 3))  # 仅前3步显示详细观测
        
        if terminated:
            print(f"\n[环境终止] 在 Step {step} 终止")
            break
    
    # 打印汇总统计
    print_separator("测试3 汇总统计")
    print(f"  总步数: {step}")
    print(f"  总奖励: {total_reward:.6f}")
    print(f"  平均奖励: {total_reward/step:.6f}")
    print(f"  总延迟: {total_delay:.6f} s")
    print(f"  总能耗: {total_energy:.6e} J")
    print(f"  总溢出任务数: {total_overflow}")


def test_action_space_validity(env: SatelliteSingleAgentEnv):
    """测试4: 动作空间有效性检查"""
    print_separator("测试4: 动作空间有效性检查")
    
    # 重置环境
    env.reset(seed=123)
    
    print(f"\n[动作空间检查]")
    print(f"  动作空间类型: {type(env.action_space)}")
    print(f"  动作空间 nvec: {env.action_space.nvec}")
    print(f"  动作空间形状: {env.action_space.shape}")
    
    # 测试随机动作采样
    print(f"\n[测试随机动作采样]")
    for i in range(5):
        random_action = env.action_space.sample()
        print(f"  随机动作 {i+1}: {random_action[:10]}... (显示前10个)")
    
    # 测试掩码后的动作采样
    print(f"\n[测试掩码动作采样]")
    mask_matrix = env._get_per_task_masks()
    for i in range(5):
        masked_action = sample_masked_action(env, mask_matrix)
        print(f"  掩码动作 {i+1}: {masked_action[:10]}... (显示前10个)")
    
    # 验证掩码动作是否都满足约束
    print(f"\n[验证掩码动作约束]")
    num_tasks = min(len(env.current_task_pool), env.I_max)
    for i in range(num_tasks):
        target = masked_action[i]
        is_valid = mask_matrix[i, target]
        print(f"  任务{i}: 选择目标{target}, 有效性={is_valid}")


def test_episode_rollout(env: SatelliteSingleAgentEnv, max_steps: int = 50):
    """测试5: 完整 episode 测试"""
    print_separator(f"测试5: 完整 Episode 测试 (最大 {max_steps} 步)")
    
    obs, info = env.reset(seed=456)
    
    step_rewards = []
    step_delays = []
    step_energies = []
    step_overflows = []
    step_tasks = []
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        step += 1
        
        # 获取掩码并采样动作
        mask_matrix = env._get_per_task_masks()
        action = sample_masked_action(env, mask_matrix)
        
        # 执行 step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 记录数据
        step_rewards.append(reward)
        step_delays.append(info.get('total_delay', 0))
        step_energies.append(info.get('total_energy', 0))
        step_overflows.append(info.get('overflow_count', 0))
        step_tasks.append(info.get('num_tasks', 0))
        
        # 每10步打印一次进度
        if step % 10 == 0:
            print(f"  Step {step}: reward={reward:.4f}, "
                  f"delay={info.get('total_delay', 0):.4f}, "
                  f"tasks={info.get('num_tasks', 0)}")
    
    # 打印汇总统计
    print(f"\n[Episode 完成]")
    print(f"  总步数: {step}")
    print(f"  终止原因: {'达到最大步数' if step >= max_steps else '环境终止'}")
    print(f"\n[奖励统计]")
    print(f"  总奖励: {sum(step_rewards):.6f}")
    print(f"  平均奖励: {np.mean(step_rewards):.6f}")
    print(f"  最大奖励: {max(step_rewards):.6f}")
    print(f"  最小奖励: {min(step_rewards):.6f}")
    
    print(f"\n[延迟统计]")
    print(f"  总延迟: {sum(step_delays):.6f} s")
    print(f"  平均延迟: {np.mean(step_delays):.6f} s")
    
    print(f"\n[能耗统计]")
    print(f"  总能耗: {sum(step_energies):.6e} J")
    print(f"  平均能耗: {np.mean(step_energies):.6e} J")
    
    print(f"\n[任务统计]")
    print(f"  总处理任务: {sum(step_tasks)}")
    print(f"  平均每步任务: {np.mean(step_tasks):.2f}")
    print(f"  总溢出任务: {sum(step_overflows)}")


def test_maskable_ppo_compatibility(env: SatelliteSingleAgentEnv):
    """测试6: MaskablePPO 兼容性检查"""
    print_separator("测试6: MaskablePPO 兼容性检查")
    
    obs, info = env.reset(seed=789)
    
    # 检查 action_masks 方法
    print(f"\n[检查 action_masks 方法]")
    masks = env.action_masks()
    print(f"  action_masks() 返回类型: {type(masks)}")
    print(f"  action_masks() 形状: {masks.shape}")
    print(f"  action_masks() dtype: {masks.dtype}")
    print(f"  期望形状: ({env.I_max * env.B},)")
    
    assert masks.shape == (env.I_max * env.B,), f"掩码形状错误: {masks.shape}"
    assert masks.dtype == bool, f"掩码 dtype 错误: {masks.dtype}"
    
    # 检查掩码重塑
    print(f"\n[检查掩码重塑]")
    mask_2d = masks.reshape(env.I_max, env.B)
    print(f"  重塑后形状: {mask_2d.shape}")
    print(f"  期望形状: ({env.I_max}, {env.B})")
    
    # 检查每个任务至少有一个有效动作
    print(f"\n[检查每个任务是否有有效动作]")
    num_tasks = min(len(env.current_task_pool), env.I_max)
    all_valid = True
    for i in range(num_tasks):
        valid_count = mask_2d[i].sum()
        if valid_count == 0:
            print(f"  警告: 任务 {i} 没有有效动作!")
            all_valid = False
        else:
            print(f"  任务 {i}: {valid_count} 个有效动作")
    
    if all_valid:
        print(f"\n[检查通过] 所有任务都有至少一个有效动作")
    else:
        print(f"\n[检查失败] 存在没有有效动作的任务")
    
    # 验证掩码动作执行
    print(f"\n[验证掩码动作执行]")
    mask_matrix = env._get_per_task_masks()
    action = sample_masked_action(env, mask_matrix)
    
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  执行掩码动作成功!")
        print(f"  返回奖励: {reward:.6f}")
    except Exception as e:
        print(f"  执行掩码动作失败: {e}")


# ======================== 主程序入口 ========================

def main():
    """主测试函数"""
    print_separator("SatelliteSingleAgentEnv 环境测试", char="*", length=80)
    print("本测试脚本用于验证环境的创建、reset 和 step 功能")
    print("为后续训练 MaskablePPO 算法做准备")
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 测试1: 创建环境
    env = test_env_creation()
    
    # 测试2: 重置环境
    test_env_reset(env)
    
    # 测试3: 单步执行（详细打印）
    # env.reset(seed=42)  # 重置环境
    test_env_step(env, num_steps=5)
    
    # 测试4: 动作空间有效性
    test_action_space_validity(env)
    
    # # 测试5: 完整 episode
    # test_episode_rollout(env, max_steps=50)
    
    # # 测试6: MaskablePPO 兼容性
    # test_maskable_ppo_compatibility(env)
    
    print_separator("所有测试完成", char="*", length=80)
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
