from typing import IO
import numpy as np
from math import atan2, atan, acos, asin, sin, cos, pi, pow, sqrt, erfc, degrees, radians, log2
from skyfield.api import EarthSatellite, load
from skyfield.toposlib import wgs84
from dataclasses import dataclass
from sympy import symbols, solve
import logging
import os
from datetime import datetime, timedelta

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go

'''
    卫星环境相关的工具函数、类,根据多智能体项目中的core.py改造,适配任务卸载环境
'''


# 配置日志记录
def setup_logger(name='satellite_env', log_level=logging.INFO):
    """
    设置日志记录器
    Args:
        name: 日志记录器名称
        log_level: 日志级别
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建logs目录
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{log_dir}/satellite_env_{timestamp}.log'
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 清除已有的处理器，避免重复
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置格式器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建全局日志记录器
logger = setup_logger()

'''
    自定义卫星和用户类
'''
# 【优化】预加载 timescale 对象，避免每次调用都重新加载
_TIMESCALE = load.timescale()

@dataclass
class Time:
    """时间类，用于卫星位置计算"""
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: float = 0.0

    def to_skyfield_time(self):
        """转换为skyfield时间对象【优化】使用全局 timescale 对象"""
        return _TIMESCALE.utc(self.year, self.month, self.day, 
                              self.hour, self.minute, self.second)

# # 服务实例类
# class ServiceInstance(object):
#     def __init__(self, id, size):
#         self.service_id = id
#         self.instance_size = size

# 任务请求实例类
class Task(object):
    def __init__(self, size, cycles, max_delay):
        self.task_size = size            # Z_i kbit
        self.computing_requirement = cycles  # k_i 单位数据所需CPU周期数
        self.delay_requirement = max_delay   # D_i 秒
    
    def total_cpu_cycles(self) -> float:
        """
        总 CPU 周期数 = Z_i * k_i
        """
        return self.task_size * self.computing_requirement

# 用户簇实体类，原服务迁移问题中的用户簇，任务卸载环境中IoT设备继承该类
class UserCluster(object):
    def __init__(self, id, lon, lat, task:Task, task_size):
        # 用户簇索引
        self.id = id
        # 用户地理位置——经度、纬度
        self.lon = lon
        self.lat = lat
        # 用户天线的最大张角
        self.up_ang = int(160)
        # self.val = val # 最小仰角
        # 当前所在卫星
        self.current_sat = None

    # def is_visible_to(self, sat, min_elevation_angle_deg):
    #     """
    #     TODO: 可见性计算参考卫星平台代码
    #     :param sat:卫星节点对象
    #     计算是否与卫星可见，根据论文公式 (4)(5)
    #     返回可见性起始时间 T_start, 可见时间 T_vis
    #     """
    #     Re = 6371e3  # 地球半径 (m)
    #     h = sat.height
    #     v = sat.velocity  # 卫星速度 (m/s)，粗略值
    #     theta = np.radians(min_elevation_angle_deg)
        
    #     gamma = np.arccos((Re / (Re + h)) * np.cos(theta)) - theta
    #     T_vis = 2 * gamma * (Re + h) / v
    #     return T_vis

class IoTDevice(UserCluster):
    """
    单个 IoT 设备（或用户簇）的扩展：
    - 继承现有 UserCluster 的地理位置信息
    - 增加本地 CPU 频率、发射功率等
    """
    def __init__(self, 
                 id: int, 
                 lon: float, 
                 lat: float,
                 lambda0: float,          # 单个设备的任务到达率 λ0
                 f_local: float,          # f_i^{loc}, 本地 CPU 频率 (Hz 或 cycles/s)
                 p_tx: float,             # P_i^{tx}, 上行发射功率 (W)
                 kappa_ue: float,         # κ_ue, 本地能耗系数 (J/cycles)
                 default_task: Task = None):
        super().__init__(id, lon, lat, task=default_task, task_size=default_task.task_size if default_task else 0)

        self.lambda0 = lambda0
        self.f_local = f_local
        self.p_tx = p_tx
        self.kappa_ue = kappa_ue

        # 当前时隙待处理任务队列（可以先做成 FIFO 队列）
        self.task_queue = []

    def generate_tasks(self, tau: float = 1.0):
        """
        确定生成的任务数量 X ~ Poisson(lambda0 * tau)
        """
        # 注意：lambda0 是单位时间(tau=1s)内的平均到达率
        num_arrivals = np.random.poisson(self.lambda0 * tau)
        
        for _ in range(num_arrivals):
            # --- 参数分布设置 (建议根据具体论文场景调整) ---
            #创建任务参数
            
            # Z_i: 任务大小，均匀分布 [500, 1000] kbit
            z_i = int(np.random.uniform(5, 10)) * 1e2  
            
            # k_i: 计算密度，均匀分布 [] Gcycles/kbit
            k_i = int(np.random.uniform(2, 10)) * 1e-4
            
            # D_i
            d_i = 3
            
            # --- 实例化任务 ---
            task = Task(size=z_i, cycles=k_i, max_delay=d_i)
            
            # 添加到本地队列和返回列表
            self.task_queue.append(task)
        

class CloudServer(UserCluster):
    """
    简化的云端计算节点，为了复用usercluster的属性和方法继承
    """
    def __init__(self, id, lon, lat, f_cloud: float):
        super().__init__(id=id, lon=lon, lat=lat, task=None, task_size=None)
        self.f_cloud = f_cloud  # f_cloud, 云端计算频率 (Gcycles/s)


    def compute_delay(self, task: Task) -> float:
        """
        计算云端计算延迟：T_cloud^{exec} = (Z_i * k_i) / f_cloud
        """
        total_cycles = task.task_size * task.computing_requirement #Gcycles
        return total_cycles / self.f_cloud

class SatelliteNode(object):
    '''卫星实体类'''
    def __init__(self, 
                 id, h, i, tran_power, tran_gain, rec_gain, 
                 tle_line1, tle_line2, 
                 comp_resource:float,   # f_s^{comp}, 卫星计算频率
                 kappa_sat: float,           # κ_sat, 卫星计算能耗系数
                 buffer_capacity: float # 队列容量上限 (Cycles)
                 ):
        # 卫星索引
        self.id = id
        # 卫星轨道半径
        self.h = h
        # 卫星轨道倾角
        self.i = i
        # 卫星天线的最大张角
        self.down_ang = int(160) 
        # 卫星计算CPU资源(Gcycles/s) 参考值100
        self.comp_resource = comp_resource
        # 卫星计算能耗系数
        self.kappa_sat = kappa_sat
        # 队列最大长度
        self.buffer_capacity = buffer_capacity
        # 发射功率
        self.tran_power = tran_power
        # 发射增益
        self.tran_gain = tran_gain
        # 接收增益
        self.rec_gain = rec_gain
        # wgs84库的卫星对象
        self.sat = EarthSatellite(tle_line1, tle_line2)
        # 可见用户簇
        self.visible_user = []
        # 执行卸载任务的IoT设备, IoTDevice类
        self.service_users = []
        # 可见目标卫星列表, SatelliteNode类
        self.target_sat_list = []
        # 【优化】位置缓存：存储 (time_key, position) 避免重复计算
        self._pos_cache_key = None
        self._pos_cache_value = None
        # 【新增】轨道速度，单位：km/s
        # self.velocity = self._compute_orbit_velocity(h)  # km/s

    # def _compute_orbit_velocity(self, orbit_altitude_km):
    #     """
    #     计算LEO卫星圆轨道速度，单位：km/s
    #     """
    #     G = 6.67430e-11      # 万有引力常数
    #     M = 5.972e24         # 地球质量
    #     R_e = 6371e3         # 地球半径 (m)
    #     R = R_e + orbit_altitude_km * 1e3
    #     v = np.sqrt(G * M / R) / 1e3  # 转为km/s
    #     return v

    def _satellite_pos(self, time: Time, pos='xyz'):
        """
        计算某时刻卫星对象的位置
        【优化】添加位置缓存，同一时刻只计算一次

        Args:
            yr, mon, day, hr, mins, sec: 年月日时分秒
            pos: 输出格式
                - 'spt': 星下点（经度、纬度、高度）输出
                - 'xyz': WGS84三维坐标（x/y/z）输出

        Return: 
            坐标列表
        """
        # 生成缓存键（使用时间和输出格式作为键）
        cache_key = (time.year, time.month, time.day, time.hour, time.minute, time.second, pos)
        
        # 检查缓存
        if self._pos_cache_key == cache_key:
            return self._pos_cache_value
        
        # 缓存未命中，进行计算
        t = time.to_skyfield_time()
        geocentric = self.sat.at(t)
        # 转化为wgs84
        wgs84_pos = wgs84.geographic_position_of(geocentric)
        lon = wgs84_pos.longitude.degrees
        lat = wgs84_pos.latitude.degrees
        alt = wgs84_pos.elevation.km
        # 按格式输出
        if pos == 'xyz':
            result = [
                (alt + wgs84.radius.km) * cos(lat/180*pi) * cos(lon/180*pi),
                (alt + wgs84.radius.km) * cos(lat/180*pi) * sin(lon/180*pi),
                (alt + wgs84.radius.km) * sin(lat/180*pi)
            ]
        else:
            result = [lon, lat, alt+wgs84.radius.km]
        
        # 更新缓存
        self._pos_cache_key = cache_key
        self._pos_cache_value = result
        
        return result 
    
    def _get_visible_user(self, users: IoTDevice, time: Time):
        '''
        计算卫星对象的可见用户族
        Args:
            users: 需要进行判断的用户族
            time: 时间
        Return:
            如果可见则返回dist,不可见返回-1
        '''
        pos_sat = self._satellite_pos(time)
        # 获取卫星的可见性阈值
        limit_val = _get_limit_elevation_ang_or_dist(
            self.h, users.up_ang, self.down_ang)
        # print("可见性阈值：", limit_val)
        dist = is_visible_or_dist(pos_sat, users.lon, users.lat, limit_val, val_type="elevation_ang")
        if dist > 0:
            return dist
        else:
            return -1

class SatMECNode(SatelliteNode):
    """
    卫星 MEC 节点：在 SatelliteNode 基础上加入队列/计算信息
    """
    def __init__(self, 
                 id, h, i, 
                 tran_power, tran_gain, rec_gain, 
                 tle_line1, tle_line2,
                 comp_resource:float,   # f_s^{comp}, 卫星计算频率
                 kappa_sat: float,           # κ_sat, 卫星计算能耗系数
                 buffer_capacity: float # 队列容量上限 (GCycles)
                 ):
        super().__init__(id, h, i, tran_power, tran_gain, rec_gain, tle_line1, tle_line2, comp_resource, kappa_sat, buffer_capacity)

        # # 当前队列积压 (GCycles)
        self.queue_backlog = 0.0
        # FIFO 任务队列：每项 (device, cycles_remaining)，用于任务完成后从 service_users 移除 device
        self.task_queue = []

    def enqueue_task(self, device, cycles: float) -> bool:
        """
        将任务入队（整包接受或整包拒绝）。
        :param device: 产生该任务的设备，用于维护 service_users
        :param cycles: 任务计算量 (Cycles)
        :return: True 表示溢出（拒绝入队），False 表示入队成功
        """
        if self.queue_backlog + cycles > self.buffer_capacity:
            return True  # 整包拒绝

        self.task_queue.append((device, cycles))
        self.queue_backlog += cycles
        if device not in self.service_users:
            self.service_users.append(device)
        return False

    def process_queue(self, duration: float) -> tuple[bool, float]:
        """
        按 FIFO 处理队列，消耗 duration 秒的计算量；任务完成后从 service_users 移除该 device（若队列中无该 device 其它任务）。
        每步每星只调用一次。
        :param duration: 时隙持续时间 (s)
        :return: (overflow_flag, processed_cycles)，溢出仅在入队时产生，此处恒为 False
        """
        service_capacity = self.comp_resource * duration
        remaining = service_capacity
        processed = 0.0
        while remaining > 0 and self.task_queue:
            device, cycles = self.task_queue[0]
            if cycles <= remaining:
                remaining -= cycles
                processed += cycles
                self.queue_backlog -= cycles
                self.task_queue.pop(0)
                if not any(d is device for d, _ in self.task_queue):
                    try:
                        self.service_users.remove(device)
                    except ValueError:
                        pass
            else:
                self.task_queue[0] = (device, cycles - remaining)
                self.queue_backlog -= remaining
                processed += remaining
                remaining = 0.0
        return False, processed

    def get_queue_delay(self):
        """
        计算排队时延：新任务到达后，需要等待当前队列积压被处理完的时间。
        应在「本任务入队前」调用，此时 queue_backlog 不包含本任务；
        若在 enqueue_task 之后调用，则排队时延会包含本任务的计算量，与计算时延重复。
        """
        if self.comp_resource <= 0:
            return float('inf')
        return self.queue_backlog / self.comp_resource



### 工具函数
# 卫星基础运行
def _get_limit_elevation_ang_or_dist(sat_h, up_ang, down_ang,
                                         output="elevation_ang"):
        """
        获取两设备间的极限值
        值的类型是最小仰角/最大距离
        计算星地链路和不同高度轨道间的星座链路使用

        Args:
            h1, h2: 两设备高度
            up_ang: 位于低处的设备向上看的最大张角
            down_ang: 位于高处的设备向下看的最大张角
            output: 输出内容，"elevation_ang"指输出最小仰角，"dist"指输出最大距离
            
        Returns:
            单位为度的最小仰角
        """
        # 模型准备
        h = sat_h
        R = 6371.393 #地球半径, 表示用户的高度
        K = h * h - R * R
        cos_2 = cos(down_ang / 360 * pi)
        cos_1 = cos(up_ang / 360 * pi)
        
        # 模型求解
        l = symbols('l', real=True)
        f1 = l * l - 2 * l * h * cos_2 + K
        f2 = l * l + 2 * l * R * cos_1 - K
        ans1 = solve([f1])  # 第一个方程的解集，可能0~2个解
        ans2 = solve([f2])  # 第二个方程的解集，有2个解
        
        # 处理解
        if len(ans1) != 2:
            l = ans2[1][l]  # 设备距离最大值
        else:
            if ans1[1][l] <= ans2[1][l]:
                l = ans2[1][l]
            else:
                l = min(ans1[0][l], ans2[1][l])
        if output == "dist":
            return l
        # 满足约束的最优值
        M = acos((h * h + R * R - l * l) / 2 / R / h)  # ∠3的最大值
        if M >= (up_ang + down_ang) / 2:
            return 90 - up_ang / 2
        else:
            return 90 - down_ang / 2 - M

# 星地链路
def is_visible_or_dist(pos_sat, lon, lat, val, val_type="elevation_ang"):
    """
    通过地面站对卫星的仰角或距离，判断卫星是否可见

    Args: 
        pos_sat: 卫星xyz位置
        lon: 地面站经度
        lat: 地面站纬度
        val: 临界值，是最小仰角或最大距离
        val_type: 值类型，"elevation_ang"指临界值最小仰角，"dist"指最大距离

    Returns:
        若不可见，返回0
        若可见，返回星地距离
    """
    earth_r = 6371.393 #地球半径
    # 经纬度转为rad
    lat = radians(lat)
    lon = radians(lon)
    # 地面站xyz坐标
    x = earth_r * cos(lat) * cos(lon)
    y = earth_r * cos(lat) * sin(lon)
    z = earth_r * sin(lat)
    # print("地面站位置", x, y, z)
    # 矢量，地面站指向卫星
    dX = pos_sat[0] - x
    dY = pos_sat[1] - y
    dZ = pos_sat[2] - z
    # 星地距离
    dist = sqrt(dX**2 + dY**2 + dZ**2)
    # 根据指标判断可见性
    if val_type == "dist":
        return dist if val >= dist else 0
    else:
        # 将矢量转换为 ENU 坐标
        t = -sin(lon) * dX + cos(lon) * dY
        n = -sin(lat) * cos(lon) * dX - sin(lat) * sin(lon) * dY + cos(lat) * dZ
        u = cos(lat) * cos(lon) * dX + cos(lat) * sin(lon) * dY + sin(lat) * dZ
        # 仰角
        alt_zeta = degrees(atan2(u, sqrt(t**2 + n**2)))
        # print("地面站仰角：", alt_zeta, "度")
        # 和最小仰角进行比较，若比它还小，说明不可见
        return dist if alt_zeta >= val else 0


# def predict_next_step_visibility(user: UserCluster, sat: Satellite, next_time: Time):
    """
    预测下一时刻用户对卫星的可见性
    Args:
        user: UserCluster对象
        sat: Satellite对象
        next_time: 下一时刻的时间对象
    Returns:
        bool: 下一时刻是否可见
    """
    # 获取下一时刻卫星位置
    next_sat_pos = sat._satellite_pos(next_time)
    
    # 使用环境中的可见性判断逻辑
    min_elev_ang = _get_limit_elevation_ang_or_dist(sat.h, user.up_ang, sat.down_ang, "elevation_ang")
    
    # 使用现有的is_visible_or_dist函数判断可见性
    visibility_result = is_visible_or_dist(next_sat_pos, user.lon, user.lat, min_elev_ang, "elevation_ang")
    
    # 如果返回距离大于0，说明可见
    return visibility_result > 0


def get_sat_dist(sat1: SatelliteNode, sat2: SatelliteNode, time: Time):
    """
    计算卫星间的距离, 若不可见则返回inf
    
    Args: 
        sat1, sat2: SatelliteNode
    
    Returns:
        卫星间的角度，不可见则返回inf
    """
    earth_r = 6371.393 #地球半径
    # # 计算向量夹角，保证acos不出错
    # pos1 = sat1._satellite_pos(time)
    # pos2 = sat2._satellite_pos(time)
    # print(f"卫星1位置", pos1, f"卫星2位置", pos2)
    # # sat1,sat2的高度应该是相同的
    # cosL = (pos1[0]*pos2[0]+pos1[1]*pos2[1]+pos1[2]*pos2[2])/sat1.h/sat2.h
    # if cosL <= -1:
    #     L = pi
    # elif cosL >= 1:
    #     L = 0
    # else:
    #     L = acos(cosL)
    # # 若被地球挡住，则不可见；否则返回两星距离
    # if sat1.h * cos(L / 2) <= earth_r:
    #     return np.inf
    # else:
    #     return 2 * sat1.h * sin(L / 2)
    pos1 = np.array(sat1._satellite_pos(time))
    pos2 = np.array(sat2._satellite_pos(time))
    # print(f"卫星1位置", pos1, f"卫星2位置", pos2)
    # 欧氏距离
    distance = np.linalg.norm(pos1 - pos2)
    
    # 取两卫星连线中点
    midpoint = 0.5 * (pos1 + pos2)
    midpoint_norm = np.linalg.norm(midpoint)
    
    # 如果中点在地球半径以内，说明连线被地球遮挡
    if midpoint_norm < earth_r:
        return np.inf
    elif distance > 3000:
        return np.inf
    else:
        return distance

def link_data_rate(sat1: SatelliteNode, sat2: SatelliteNode, d, time: Time):
    """计算链路速率（根据公式(9)(10)）"""
    c = 3e8  # 光速
    k = 1.38e-23  # Boltzmann常数
    Un_dBK = 25  # 系统噪声温度（dBK）
    EbN0 = 1  # 接收能量/噪声谱密度
    A_dB = 1.5  # 链路裕度（dB）
    carrier_freq = 23e9  # 载波频率 Hz

    # dBK、dB转线性
    Un = 10 ** (Un_dBK / 10)  # K
    A = 10 ** (A_dB / 10)     # 无单位

    Gt = 10 ** (sat1.tran_gain / 10)
    Gr = 10 ** (sat2.rec_gain / 10)

    if d == np.inf:
        return 0
    else:
        # 距离单位转换为m
        d_m = d * 1000  # km to m
        # 自由空间路径损耗
        Lfs = (c / (4 * np.pi * d_m * carrier_freq)) ** 2
        # 链路速率计算
        R = (sat1.tran_power * Gt * Gr * Lfs) / (k * Un * EbN0 * A)
        return R

class Walker(object):
    """
    Walker星座类，用于创建卫星星座
    """
    def __init__(self, num_sats, h, angle, P_num,  
                 sat_tran_power:list, sat_tran_gain:list, sat_rec_gain:list, 
                 sat_comp_resource:list, sat_kappa_sat:list, sat_buffer_capacity:list):
        self.num_sats = num_sats
        self.h = h
        self.angle = angle
        self.P_num = P_num
        self.sat_comp_resource = sat_comp_resource
        self.sat_tran_power = sat_tran_power
        self.sat_tran_gain = sat_tran_gain
        self.sat_rec_gain = sat_rec_gain
        self.sat_kappa_sat = sat_kappa_sat
        self.sat_buffer_capacity = sat_buffer_capacity

    def _generate_tles_line2(self, N, h, i, P, F=int(1)):
        """
        生成walker星座中所有卫星的TLE星历的第二行
        Args:
            N: int，walker星座中卫星总数
            h: float，卫星轨道高度，单位km
            i: float，卫星轨道倾角，单位度
            P: int，walker星座的轨道面数
            F: int，walker星座的相位数，默认为1
        Returns:
            list，包含所有卫星的第二行TLE
            STARLINK-1010
            1 44716U 19074D   25187.23278464  .00110409  00000+0  17717-2 0  9991
            2 44716  53.0543  195.3554  0010293 348.5237  11.5536 15.52174921312007
                    轨道倾角 升交点赤经 轨道偏心率 升交点角距 平近点角 每日平均运动
        """
        GM = 3986005 * 10 ** 8      # 【WGS-84】地球引力和地球质量的乘积
        def _tle_format(num, all=8, dec=4):
            return str(f"%.{dec}f"%num).zfill(all)
        
        tles = []                     # 计算各卫星tle，并加入该列表
        detu = 360 / N * F  # 邻轨对应卫星间的相位差
        # walker星座各卫星每天绕地圈数
        circles = sqrt(GM) * 12 * 3600 / pi / pow(h*1000, 1.5)
        num_S = int(N/P)  # 每个轨道面上的卫星数

        for sat_id in range(N):
            Pm = int(sat_id / num_S)  # 轨道面编号，0 ~ P-1
            Nm = sat_id % num_S       # 轨道内编号，0 ~ S-1
            omega_m = 90 / P * Pm          # 升交点赤经 omega_m = 180 / P * Pm
            # 测试生成附近的几颗卫星
            u_m = 0 + (45 / num_S * Nm) % 360 + detu * Pm  # 平近点角 u_m = 360 / num_S * Nm + detu * Pm
            tles.append(
                f'2 44716 {_tle_format(i)} {_tle_format(omega_m)} 0000000 000.0000 '
                f'{_tle_format(u_m)} {_tle_format(circles,11,8)}'
            )
        return tles
        # 如果需要以STARLINK-1010为基准生成相邻的卫星tle
        # base_mean_anomaly = 11.5536
        # num_sats = 5
        # mean_motion = 15.52174921
        # inclination = 53.0543
        # raan = 195.3554
        # ecc = 0.0010293
        # arg_perigee = 348.5237

        # tles = []
        # for i in range(num_sats):
        #     mean_anomaly = (base_mean_anomaly + i * (360/num_sats)) % 360
        #     tle_line2 = (
        #         f"2 44716 "
        #         f"{inclination:8.4f} "
        #         f"{raan:8.4f} "
        #         f"{ecc*1e7:07.0f} "
        #         f"{arg_perigee:8.4f} "
        #         f"{mean_anomaly:8.4f} "
        #         f"{mean_motion:11.8f} 99999"
        #     )
        #     tles.append(tle_line2)

    def create_satellites(self):
        """
        创建卫星星座并返回SatMECNode对象列表
        
        Returns:
            list: SatMECNode对象列表
        """
        tle_list_line2 = self._generate_tles_line2(self.num_sats, self.h, self.angle, self.P_num)
        satellite_list = []

        for i in range(self.num_sats):
            # 生成TLE数据
            tle_line1 = f"1 44716U 19074D   25187.23278464  .00110409  00000+0  17717-2 0  9991"
            tle_line2 = tle_list_line2[i]
            #print(f"Creating Satellite {i} with TLE:\n{tle_line1}\n{tle_line2}")

            # 创建SatMECNode对象，注意参数顺序要与SatMECNode.__init__匹配
            # SatMECNode.__init__参数顺序: id, h, i, tran_power, tran_gain, rec_gain, 
            #                              tle_line1, tle_line2, comp_resource, kappa_sat, buffer_capacity
            sat = SatMECNode(
                id=i,
                h=self.h,
                i=self.angle,
                tran_power=self.sat_tran_power[i],
                tran_gain=self.sat_tran_gain[i],
                rec_gain=self.sat_rec_gain[i],
                tle_line1=tle_line1,
                tle_line2=tle_line2,
                comp_resource=self.sat_comp_resource[i],
                kappa_sat=self.sat_kappa_sat[i],
                buffer_capacity=self.sat_buffer_capacity[i]
            )
            satellite_list.append(sat)


        return satellite_list
    
    def _update_sat_links(self, time: Time, satellites, sat_topology, sat_links):
        """
        Args:
        time: 当前时间
        satellites: 卫星列表:Satellite对象列表
        sat_topology: 卫星拓扑结构，字典形式
        sat_links: 卫星链路信息，字典形式
        """
        """
        采用grid结构：
        - 同轨相邻卫星连接
        - 邻轨相邻卫星连接
        更新self.sat_topology, self.sat_links
        """
        S = int(self.num_sats / self.P_num)  # 每个轨道面上的卫星数
        sat_topology.clear()
        sat_links.clear()

        for i, sat in enumerate(satellites):
            sat.target_sat_list.clear()
            sat_topology[sat.id] = []

        # 同轨相邻连接
        # print("[DEBUG] 开始同轨相邻连接...")
        for p in range(self.P_num):
            for s in range(S):
                a_idx = p * S + s
                b_idx = p * S + (s + 1) % S
                sat1 = satellites[a_idx]
                sat2 = satellites[b_idx]
                dist = get_sat_dist(sat1, sat2, time) #卫星必须满足可见关系
                if dist == np.inf:
                    continue
                rate = link_data_rate(sat1, sat2, dist, time)
                #print(f"[DEBUG] 同轨连接: sat{sat1.id}(轨道{p},位置{s},索引{a_idx}) <-> sat{sat2.id}(轨道{p},位置{(s+1)%S},索引{b_idx})")
                self._add_link(sat1, sat2, dist, rate, sat_topology, sat_links)

        # 邻轨相邻连接
        # print("[DEBUG] 开始邻轨相邻连接...")
        if self.P_num > 1:
            for p in range(self.P_num):
                for s in range(S):
                    a_idx = p * S + s
                    b_p = (p + 1) % self.P_num
                    b_idx = b_p * S + s
                    sat1 = satellites[a_idx]
                    sat2 = satellites[b_idx]
                    dist = get_sat_dist(sat1, sat2, time) #卫星必须满足可见关系
                    if dist == np.inf:
                        continue
                    rate = link_data_rate(sat1, sat2, dist, time)
                    #print(f"[DEBUG] 邻轨连接: sat{sat1.id}(轨道{p},位置{s},索引{a_idx}) <-> sat{sat2.id}(轨道{b_p},位置{s},索引{b_idx})")
                    self._add_link(sat1, sat2, dist, rate, sat_topology, sat_links)

        # print("卫星间grid连接关系", sat_topology)
        # print("卫星间grid链路信息", sat_links)
    
    def _add_link(self, sat1, sat2, dist, rate, sat_topology, sat_links):
        """
        将sat1和sat2的双向连接加入拓扑
        """
        # print(f"[DEBUG] _add_link 被调用: sat{sat1.id} <-> sat{sat2.id}, 距离={dist:.2f}, 速率={rate:.2e}")
        
        # 避免自己连接到自己
        if sat1.id == sat2.id:
            #print(f"[警告] 尝试连接卫星到自己: sat{sat1.id}")
            return
            
        # 避免重复连接
        if sat2.id in sat_topology[sat1.id]:
            # print(f"[警告] 重复连接: sat{sat1.id} -> sat{sat2.id}")
            # print(f"[DEBUG] sat{sat1.id} 当前连接列表: {sat_topology[sat1.id]}")
            return
            
        if sat1.id in sat_topology[sat2.id]:
            # print(f"[警告] 重复连接: sat{sat2.id} -> sat{sat1.id}")
            # print(f"[DEBUG] sat{sat2.id} 当前连接列表: {sat_topology[sat2.id]}")
            return
        
        # 添加连接
        sat_topology[sat1.id].append(sat2.id)
        sat_links[(sat1.id, sat2.id)] = {
            "distance": dist,
            "data_rate": rate
        }
        sat1.target_sat_list.append(sat2)

        sat_topology[sat2.id].append(sat1.id)
        sat_links[(sat2.id, sat1.id)] = {
            "distance": dist,
            "data_rate": rate
        }
        sat2.target_sat_list.append(sat1)
    
    

# 卫星世界类
class SatelliteWorld(object):
    """
    参考mpe环境的World类
    """
    def __init__(self, walker:Walker, time: Time):
        '''环境相关属性'''
        # walker星座对象
        self.walker = walker
        # 卫星列表，元素是Satellite对象
        self.satellites = []
        # 用户簇列表，元素是IoTDevice对象
        self.user_clusters = []
        # 云数据中心
        self.cloud = None

        # 最大时间步——在创建world时用脚本中的episode_length赋值
        self.world_length = 100
        # 当前时间步
        self.world_step = 0
        # 智能体数量
        self.num_agents = 0
        # 物理世界的时间，对应年月日
        # self.time = 0  # 删除float类型
        # 时间步长，也就是world_step一步对应的物理世界的时间
        self.dt = 2.0
        # 新增：当前物理世界的时间对象
        self.current_time = time
        # 创建初始时间的深拷贝，避免引用问题
        self.initial_time = Time(
            year=time.year,
            month=time.month,
            day=time.day,
            hour=time.hour,
            minute=time.minute,
            second=time.second
        )

        # # 兼容MPE环境的属性
        # self.dim_c = 0  # 通信维度，卫星环境暂时不需要
        # self.dim_p = 3  # 位置维度，卫星环境为3D

        '''拓扑更新'''
        # 邻接表结构，表示卫星与其他卫星的连接关系，例如 {0: [1, 2], 1: [0, 2, 3], ...}
        self.sat_topology = {}
        # 具体链路信息 ,例如{(i, j): {"distance": ..., "data_rate": ..., "visible_time": ...}}
        self.sat_links = {}
        # 用户-卫星链路信息，例如 {(user_id, sat_id): 距离，若不可见为-1}
        self.user_sat_visibility = {}

        # 【新增】可见时间管理dict
        self.visible_time = {}  # (user.id, sat.id): 可见时间秒

        #新增 服务失败次数
        self.service_failure_count = int(0)

    

    
    def _update_link_states(self, time: Time):
        """
        封装
        更新全局的卫星间链路状态，包括距离、可见时间和链路速率等信息。
        更新self.sat_topology和self.sat_links
        更新Satellite类的target_sat属性
        """
        
        # 清空现有拓扑和链路信息，确保每次调用都是全新的
        self.sat_topology.clear()
        self.sat_links.clear()
        for sat in self.satellites:
            sat.target_sat_list.clear()
            
        self.walker._update_sat_links(self.current_time, self.satellites, self.sat_topology, self.sat_links)
    
    
    # 【新增】物理世界总秒数计算（仅天、时、分、秒，可进一步支持月/年）
    def get_world_time_in_seconds(self, t: Time) -> int:
        """将Time对象转为自环境初始以来的总秒数。可自行扩展跨月/年。"""
        return (t.day * 86400 + t.hour * 3600 + t.minute * 60 + int(t.second))
    
    def _update_visibility_matrix(self, time: Time):
        '''
        更新用户和卫星的可见性
        更新self.user_sat_visibility和sat.visible_user
        '''
        now_seconds = self.get_world_time_in_seconds(time)

        self.user_sat_visibility.clear()
        for sat in self.satellites:
            sat.visible_user.clear()
            for user in self.user_clusters:
                dist = sat._get_visible_user(user, time)
                #打印调试信息
                # print("计算可见性 user{} and sat{}: {}".format(user.id, sat.id, dist))
                # 更新全局的用户-卫星链路信息-可见性和距离，不可见则dist=-1
                self.user_sat_visibility[(user.id, sat.id)] = dist
                # # ----------【新增】T_start记录逻辑 ----------
                # key = (user.id, sat.id)
                # if dist > 0:
                #     # t_rem, t_vis = self.compute_remaining_visibility_time(sat, user, self.current_time)
                #     # 上一时刻不可见，当前可见，记录T_start
                #     if key not in self.user_sat_t_start or self.user_sat_t_start[key] is None:
                #         self.user_sat_t_start[key] = now_seconds
                # else:
                #     # 当前不可见，清除T_start
                #     self.user_sat_t_start[key] = None
                # # -----------------------------------------
                # 如果可见，则更新卫星的visible_user列表中
                if dist > 0:
                    sat.visible_user.append(user)
            # print(f"卫星{sat.id}的可见用户列表{sat.visible_user}")
        # print("用户-卫星可见性信息", self.user_sat_visibility)
        #logger.info(f"用户-卫星可见性信息: {self.user_sat_visibility}")
        #logger.info(f"用户-卫星剩余可见时间: {self.visible_time}")
        # print("用户-卫星剩余可见时间: ", t_rem)
    
    def _update_cloud_current_sat(self, cloud: CloudServer):
        '''
        更新云端当前服务卫星
        '''
        best_sat = None
        best_dist = float("inf")
        for s in self.satellites:
            d = self.user_sat_visibility.get((cloud.id, s.id), -1)
            if d > 0 and d < best_dist:
                best_dist = d
                best_sat = s
            cloud.current_sat = best_sat

    def _update_device_current_sat(self, device: IoTDevice):
        """
         将设备当前卫星设为可见且距离最近的卫星（卸载到云端时用于计算上行等）
        """
        best_sat = None
        best_dist = float("inf")
        for s in self.satellites:
            d = self.user_sat_visibility.get((device.id, s.id), -1)
            if d > 0 and d < best_dist:
                best_dist = d
                best_sat = s
        # if best_sat is None:
        #     print(f"[错误] 用户{device.id}的当前最近卫星为None")
        return best_sat

    '''
        卸载到本地设备时的延迟和能耗计算
    '''
    def compute_local_delay(self, device: IoTDevice, task: Task) -> float:
        """
        
        T_i^{loc} = (Z_i * k_i) / f_i^{loc}
        """

        total_cycles = task.task_size * task.computing_requirement
        return total_cycles / device.f_local


    def compute_local_energy(self, device: IoTDevice, task: Task) -> float:
        """
        E_i^{loc} = κ_ue * (f_i^{loc})^2 * (Z_i * k_i) 计算能耗
        """
        total_cycles = task.task_size * task.computing_requirement * 1e9 #cycles
        f_local = device.f_local * 1e9
        return device.kappa_ue * (f_local ** 2) * total_cycles#cycles

    
    '''
    卸载到卫星边缘计算时的延迟和能耗计算
    '''
    def _transmission_rate(self, device: UserCluster, task: Task, sat: SatMECNode):
        # # # 获取链路信息
        # if device.current_sat is None:
        #     print(f"[错误] 用户{device.id}的current_sat为None，无法计算通信延迟")
        #     return float('inf')
        # 获取链路信息
        d_us = self.user_sat_visibility.get((device.id, sat.id))
        if d_us == -1:
            print(f"[ERROR!] 用户{device.id}到卫星{sat.id}的距离为-1")
            return 0

        # # 2. 获取参数
        # D_u = task.task_size  # 上传数据量，单位kbit
        c = 3e8  # 光速 m/s

        # 3. 信道参数 - 修正为更合理的值
        B_ui = 8  # MHz，用户到卫星的通信带宽
        P_u = 1     # W，用户发射功率
        # 修正信道增益系数，考虑自由空间路径损耗
        # 自由空间路径损耗: L = (4πd/λ)^2，其中λ = c/f，f ≈ 2GHz
        f_carrier = 2e9  # 载波频率 2GHz
        wavelength = c / f_carrier  # 波长
        # 距离单位转换为m
        d_us_m = d_us * 1000  # km to m
        # 自由空间路径损耗
        path_loss = (4 * np.pi * d_us_m / wavelength) ** 2
        # 假设用户终端天线增益 G_t_dBi = 10 dBi, 卫星天线增益 G_r_dBi = 41 dBi
        G_t = 10**(5 / 10)  # 转换为线性值
        G_r = 10**(20 / 10)  # 转换为线性值
        # 信道增益 = G_t * G_r/路径损耗
        h_ui = G_t * G_r / path_loss
        
        k = 1.38e-23  # 玻尔兹曼常数 J/K
        T = 290      # 系统噪声温度 K (一个常用参考值)
        N0 = k * T   # W/Hz,噪声功率谱密度
        # N0 = 1e-9   # W/Hz，噪声功率谱密度

        # 4. Shannon容量公式
        # 注意带宽单位需统一，假设D_u单位为Mbit，B_ui单位为MHz，需转为bit/s和Hz
        B_ui_Hz = B_ui * 1e6  # Hz，将带宽从MHz转换为Hz
        SNR = (P_u * h_ui) / (N0 * B_ui_Hz)  # 信噪比
        R_ui = B_ui_Hz * log2(1 + SNR)  # bit/s，根据香农公式计算的理论传输速率
        return R_ui

    def _compute_communication_delay(self, device: UserCluster, task: Task, sat: SatMECNode):
        """
        计算用户设备到卫星sat的通信延迟
        """
        D_u = task.task_size  # kbit
        R_ui = self._transmission_rate(device, task, sat)
        if R_ui == 0:
            return float('inf')

        # 获取链路距离
        d_us = self.user_sat_visibility.get((device.id, sat.id))

        #计算传输延迟+传播延迟
        D_u_bit = D_u * 1e3  # bit，将数据量从kbit转换为bit
        # 传播延迟：距离单位需要统一为m
        # 距离单位转换为m
        d_us_m = d_us * 1000  # km to m
        c = 3e8  # 光速 m/s
        propagation_delay = d_us_m / c  # s
        # 传输延迟
        transmission_delay = D_u_bit / R_ui  # s
        comm_delay = transmission_delay + propagation_delay  # s，总通信延迟 = 传输延迟 + 传播延迟
        
        # print("[DEBUG] 计算通信延迟: 上传数据量={}kbit, 链路速率={:.2e}bit/s, 距离={}km, 传输延迟={:.6f}s, 传播延迟={:.6f}s, 总延迟={:.6f}s".format(
        #     D_u, R_ui, d_us, transmission_delay, propagation_delay, comm_delay))

        return comm_delay

    def compute_edge_delay(self, device:IoTDevice, task: Task,
                       sat: SatMECNode,
                       c: float = 3e8) -> float:
        """
        通信延迟 + 排队延迟 + 计算延迟
        T_{i,s}^{edge} = Z_i / R_{i,s}^{up} + d_{i,s}/c + T_s^{queue} + (Z_i * k_i) / f_s^{comp}
        - R_up: 上行速率 (bit/s)
        - distance_km: 星地距离 (km)
        """
        total_cycles = task.task_size * task.computing_requirement
        # 通信延迟
        T_comm = self._compute_communication_delay(device, task, sat)
        # 排队延迟
        T_queue = sat.get_queue_delay()
        # 计算延迟
        f_comp = sat.comp_resource / len(sat.service_users)
        T_comp = total_cycles / f_comp
        # print(f"设备{device.id}, 卫星{sat.id}, 通信延迟: {T_comm:.2f}s, 排队延迟: {T_queue:.2f}s, 计算延迟: {T_comp:.2f}s")

        return T_comm + T_queue + T_comp


    def compute_edge_energy(self, device: IoTDevice, task: Task,
                            sat: SatMECNode) -> float:
        """
        通信能耗 + 卫星计算能耗:
        E_{i,s}^{trans} = P_i^{tx} * Z_i / R_{i,s}^{up} 发射功率 任务大小 上行传输速率
        E_{i,s}^{comp}  = κ_sat * (f_s^{comp})^2 * (Z_i * k_i) 能耗因子 计算资源 任务所需cycles
        """
        Z_kbits = task.task_size
        total_cycles = Z_kbits * task.computing_requirement * 1e9#cycles

        R_up = self._transmission_rate(device, task, sat)
        if R_up <= 0:
            E_tx = float('inf')
        else:
            E_tx = device.p_tx * Z_kbits * 1e3 / R_up

        f_comp = sat.comp_resource / len(sat.service_users) * 1e9
        E_comp = sat.kappa_sat * ((f_comp) ** 2) * total_cycles  #cycles
        # print(f"计算资源: {f_comp:.2e}cycles/s, 总CPU周期数: {total_cycles:.2e}cycles, 能耗系数: {sat.kappa_sat:.2e}J/cycles")
        # print(f"设备{device.id}, 卫星{sat.id}, 通信能耗: {E_tx:.2f}J, 计算能耗: {E_comp:.2f}J")
        return E_tx + E_comp



    '''
    卸载到云端数据中心的延迟和能耗
    '''
    def get_ISL_path(self, sat1: SatMECNode, sat2: SatMECNode) -> list:
        """
        同轨线性拓扑下按编号顺序给出星间转发路径。
        假设卫星在同一轨道上、节点线性连接，路径为连续编号序列。
        例如：卫星 id=1 到 id=3 的路径为 [1, 2, 3]；id=3 到 id=1 为 [3, 2, 1]。

        Args:
            sat1: 路径起点卫星
            sat2: 路径终点卫星
        Returns:
            list: 卫星 id 序列 [sat1.id, ..., sat2.id]
        """
        a, b = sat1.id, sat2.id
        if a <= b:
            return list(range(a, b + 1))
        else:
            return list(range(a, b - 1, -1))
    
    def get_backhaul_distance(self, device: IoTDevice, cloud: CloudServer, best_sat: SatMECNode) -> float:
        """
        计算设备到云中心的距离
        """
        # 将设备当前卫星设为可见且距离最近的卫星（用于计算上行等）
        sat1 = best_sat
        
        path_sat_ids = self.get_ISL_path(sat1, cloud.current_sat)
        total = 0.0
        for u, v in zip(path_sat_ids[:-1], path_sat_ids[1:]):
            link = self.sat_links.get((u, v))
            if link is None:
                return float('inf')
            total += link["distance"]
        return total


    def compute_backhaul_delay(self, device: IoTDevice, task: Task,
                               sat_in: SatMECNode, sat_out: SatMECNode,
                               T_switch: float = 10e-3,
                               c: float = 3e8) -> float:
        """
        ISL 回传延迟：
        T_ISL^{backhaul} = sum_{(u,v)∈P} [ Z_i / R_ISL + d_{u,v}/c + T^{switch} ]

        Args:
            device: 发起任务的设备（未直接用于计算，保留接口一致性）
            task: 任务
            sat_in: 入口卫星（接入星）
            sat_out: 出口卫星（连接地面站/网关的星）
            R_ISL_default: 链路未给出 data_rate 时的默认速率 (bit/s)
            T_switch: 星上交换/处理延迟 (s)
            c: 光速 (m/s)
        Returns:
            float: 回传延迟 (s)，不可达时返回 inf
        """
        path_sat_ids = self.get_ISL_path(sat_in, sat_out)
        if len(path_sat_ids) < 2:
            return 0.0  # 同一节点不存在ISL回传延迟
        Z_bits = task.task_size
        total = 0.0
        for u, v in zip(path_sat_ids[:-1], path_sat_ids[1:]):
            link = self.sat_links.get((u, v))
            if link is None:
                return float('inf')
            d_km = link["distance"]
            R_ISL = link.get("data_rate")
            if R_ISL <= 0:
                return float('inf')
            T_tx = Z_bits / R_ISL
            T_prop = d_km * 1000.0 / c
            total += T_tx + T_prop + T_switch
        return total

    def compute_cloud_total_delay(self, device: IoTDevice, task: Task, cloud: CloudServer,
                              ) -> float:
        """
        卸载到云数据中心节点的总体延迟
        T_{i,cloud} = T_{i,s_in}^{access} + T_ISL^{backhaul} + T_{s_out,G}^{down} + T_{cloud}^{exec}
        """
        Z_bits = task.task_size
        total_cycles = Z_bits * task.computing_requirement
        #接入卫星通信延迟
        sat_in = device.current_sat
        T_access = self._compute_communication_delay(device, task, sat_in)
        #ISL回传延迟
        sat_out = cloud.current_sat
        T_backhaul = self.compute_backhaul_delay(device, task, sat_in, sat_out)
        # 下行通信延迟
        T_down = self._compute_communication_delay(cloud, task, sat_out)
        # 计算延迟
        T_exec = total_cycles / cloud.f_cloud
        # print(f"设备{device.id}, 云端接入延迟: {T_access:.2f}s, ISL回传延迟: {T_backhaul:.2f}s, 下行延迟: {T_down:.2f}s, 计算延迟: {T_exec:.2f}s")
        return T_access + T_backhaul + T_down + T_exec


    def compute_cloud_energy(self, device: IoTDevice,
                             task: Task, cloud: CloudServer,
                             P_ISL: float = 1.0,
                             ) -> float:
        """
        总能耗 = 上行数据传输能耗 + 卫星中继能耗
        多跳 ISL 中继能耗：
        E_relay = Σ_{(u,v)∈P} P_ISL * Z_i / R_{u,v}
        Args:
            task: 任务（Z_i 使用 task.task_size，单位需与速率一致：bit / (bit/s)）
            P_ISL: ISL 发射功率（W），此处简化为各跳相同
            R_ISL_default: 当链路未提供 data_rate 时的默认 ISL 速率 (bit/s)
        Returns:
            float: 总能耗（J）
        """
        path_sat_ids = self.get_ISL_path(device.current_sat, cloud.current_sat)

        Z_bits = task.task_size
        backhaul_energy = 0.0
        if len(path_sat_ids) < 2:
            backhaul_energy = 0.0
        else:
            for u, v in zip(path_sat_ids[:-1], path_sat_ids[1:]):
                link = self.sat_links.get((u, v))
                if link is None:
                    return float('inf')
                R_ISL = link.get("data_rate")
                if R_ISL <= 0:
                    return float('inf')
                backhaul_energy += P_ISL * Z_bits * 1e3 / R_ISL

        R_up = self._transmission_rate(device, task, device.current_sat)
        if R_up <= 0:
            E_tx = float('inf')
        else:
            E_tx = device.p_tx * Z_bits *1e3 / R_up
        # print(f"设备{device.id}, 云端上行能耗: {E_tx:.2f}J, ISL能耗: {backhaul_energy:.2f}J")
        return E_tx + backhaul_energy



    # def get_user_service_status(self, agent: Satellite):
    #     """
    #     判断卫星agent上所有用户的服务状态
    #     Args:
    #         agent: Satellite对象，需要判断服务状态的卫星
    #     Returns:
    #         dict: 用户ID到服务状态的映射，True表示服务成功，False表示服务失败
    #     """
    #     service_status_dict = {}
        
    #     # 遍历卫星服务的所有用户
    #     for user in agent.service_users:
    #         # 检查用户是否有当前服务卫星
    #         if user.current_sat is None:
    #             print(f"[服务状态] 用户{user.id}没有分配服务卫星，服务失败")
    #             service_status_dict[user.id] = False
    #             continue
            
    #         # 检查用户与当前服务卫星的可见性
    #         visibility_key = (user.id, user.current_sat.id)
    #         if visibility_key not in self.user_sat_visibility:
    #             print(f"[服务状态] 用户{user.id}与卫星{user.current_sat.id}的可见性信息不存在，服务失败")
    #             service_status_dict[user.id] = False
    #             continue
            
    #         dist = self.user_sat_visibility[visibility_key]
    #         if dist == -1:
    #             print(f"[服务状态] 用户{user.id}与卫星{user.current_sat.id}不可见(dist={dist})，服务失败")
    #             service_status_dict[user.id] = False
    #             continue
            
    #         # # 检查卫星是否有足够的计算资源
    #         # if user.current_sat.comp_resource < user.service_instance.instance_size:
    #         #     print(f"[服务状态] 卫星{user.current_sat.id}资源不足(需要{user.service_instance.instance_size}，剩余{user.current_sat.comp_resource})，服务失败")
    #         #     service_status_dict[user.id] = False
    #         #     continue
            
    #         print(f"[服务状态] 用户{user.id}服务成功，当前服务卫星{user.current_sat.id}，距离{dist:.2f}km")
    #         service_status_dict[user.id] = True
        
    #     return service_status_dict


    # def _sats_migration_cost(self, sat: Satellite, target_sat: Satellite):
    #     """
    #     计算卫星迁移成本
    #     Args:
    #         sat: Satellite对象，源卫星
    #         target_sat: Satellite对象，目标卫星
    #     Returns:
    #         float: 迁移成本
    #     """
    #     migration_cost = 0.0
    #     c = 3e8
    #     rate = self.sat_links[(sat.id, target_sat.id)]["data_rate"]
    #     dist = self.sat_links[(sat.id, target_sat.id)]["distance"]
    #     if rate > 0:
    #         prop_delay = dist * 1000 / c
    #         bw_cost = 10e9 / rate
    #         migration_cost = 0.5*prop_delay + 0.5*bw_cost  
    #     return migration_cost
    
    def _future_dist(self, user, sat):
        visibility_vector = []
        for i in range(3):
            t = i * self.dt
            
            # 计算总秒数
            total_seconds = self.current_time.second + t
            
            # 处理时间进位
            extra_minutes = int(total_seconds // 60)
            final_seconds = total_seconds % 60
            
            extra_hours = int((self.current_time.minute + extra_minutes) // 60)
            final_minutes = (self.current_time.minute + extra_minutes) % 60
            
            # 创建新的时间对象，基于当前时间
            future_time = Time(
                self.current_time.year,
                self.current_time.month,
                self.current_time.day,
                self.current_time.hour + extra_hours,
                final_minutes,
                final_seconds
            )
            
            dist = sat._get_visible_user(user, future_time)
            visibility_vector.append(dist if dist > 0 else 0.0)
        return visibility_vector
    
    

    
    def plot_step_positions(self, step_idx, save_dir="satellite_steps"):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 1. 绘制地球球体
        r = 6371  # 地球半径，单位km
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_surface(x, y, z, color='deepskyblue', alpha=0.3)

        # 2. 绘制所有卫星
        sat_x, sat_y, sat_z = [], [], []
        for sat in self.satellites:
            pos = sat._satellite_pos(self.current_time)  # 获取当前step卫星位置
            sat_x.append(pos[0])
            sat_y.append(pos[1])
            sat_z.append(pos[2])
            ax.text(pos[0], pos[1], pos[2], f"S{sat.id}", fontsize=8, color='red')  # 标注卫星编号
        ax.scatter(sat_x, sat_y, sat_z, c='red', marker='o', label='Satellites')

        # 3. 绘制所有用户
        user_x, user_y, user_z = [], [], []
        for user in self.user_clusters:
            # 需实现经纬度到xyz的转换
            lon, lat = user.lon, user.lat
            pos = [
                r * np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                r * np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                r * np.sin(np.radians(lat))
            ]
            user_x.append(pos[0])
            user_y.append(pos[1])
            user_z.append(pos[2])
            ax.text(pos[0], pos[1], pos[2], f"U{user.id}", fontsize=8, color='green')  # 标注用户编号
        ax.scatter(user_x, user_y, user_z, c='green', marker='^', label='Users')

        # 设置视角
        ax.view_init(elev=30, azim=60)

        ax.set_title(f"Step {step_idx} 卫星与用户三维分布")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.legend()

        plt.savefig(os.path.join(save_dir, f"satellite_step_{step_idx}.png"))
        plt.close()

    

    def plot_step_positions_interactive(self, step_idx, save_dir="satellite_steps_html"):
        """
        使用 Plotly 绘制卫星与用户三维分布，并保存为可交互 HTML 图像。

        Args:
            step_idx: 当前时间步编号
            save_dir: HTML 文件保存路径
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        r = 6371  # 地球半径

        # ---------------- 地球球体 ---------------- #
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)

        surface = go.Surface(
            x=x, y=y, z=z,
            colorscale='Blues',
            opacity=0.5,
            showscale=False,
            name="Earth"
        )

        # ---------------- 卫星 ---------------- #
        sat_x, sat_y, sat_z, sat_labels = [], [], [], []
        sat_positions = {}  # 存储卫星位置，用于绘制连接线
        for idx, sat in enumerate(self.satellites):
            pos = sat._satellite_pos(self.current_time)
            sat_x.append(pos[0])
            sat_y.append(pos[1])
            sat_z.append(pos[2])
            sat_labels.append(f"S{idx}")
            sat_positions[sat.id] = pos

        sat_trace = go.Scatter3d(
            x=sat_x, y=sat_y, z=sat_z,
            mode='markers+text',
            marker=dict(size=4, color='red'),
            text=sat_labels,
            textposition="top center",
            name="Satellites"
        )

        # ---------------- 用户 ---------------- #
        user_x, user_y, user_z, user_labels = [], [], [], []
        user_positions = {}  # 存储用户位置，用于绘制连接线
        for idx, user in enumerate(self.user_clusters):
            lon, lat = user.lon, user.lat
            pos = [
                r * np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                r * np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                r * np.sin(np.radians(lat))
            ]
            user_x.append(pos[0])
            user_y.append(pos[1])
            user_z.append(pos[2])
            user_labels.append(f"U{idx}")
            user_positions[user.id] = pos

        user_trace = go.Scatter3d(
            x=user_x, y=user_y, z=user_z,
            mode='markers+text',
            marker=dict(size=5, color='green', symbol='diamond'),
            text=user_labels,
            textposition="top center",
            name="Users"
        )

        # ---------------- 卫星间连接线 ---------------- #
        sat_links_traces = []
        for sat1_id, connected_sats in self.sat_topology.items():
            if sat1_id in sat_positions:
                sat1_pos = sat_positions[sat1_id]
                for sat2_id in connected_sats:
                    if sat2_id in sat_positions and sat1_id < sat2_id:  # 避免重复绘制
                        sat2_pos = sat_positions[sat2_id]
                        
                            
                        link_trace = go.Scatter3d(
                            x=[sat1_pos[0], sat2_pos[0]],
                            y=[sat1_pos[1], sat2_pos[1]],
                            z=[sat1_pos[2], sat2_pos[2]],
                            mode='lines',
                            line=dict(color='blue', width=2, dash='dash'),
                            showlegend=False,
                            name=f"Sat-Sat Link"
                        )
                        sat_links_traces.append(link_trace)

        # ---------------- 用户-卫星连接线 ---------------- #
        user_sat_links_traces = []
        for user in self.user_clusters:
            if user.current_sat is not None and user.id in user_positions and user.current_sat.id in sat_positions:
                user_pos = user_positions[user.id]
                sat_pos = sat_positions[user.current_sat.id]
                link_trace = go.Scatter3d(
                    x=[user_pos[0], sat_pos[0]],
                    y=[user_pos[1], sat_pos[1]],
                    z=[user_pos[2], sat_pos[2]],
                    mode='lines',
                    line=dict(color='red', width=5, dash='solid'),  # 当前服务连接用红色粗实线
                    showlegend=False,
                    name=f"User-Sat Service Link"
                )
                user_sat_links_traces.append(link_trace)

        # ---------------- 可见性连接线（虚线） ---------------- #
        visibility_links_traces = []
        for user in self.user_clusters:
            if user.id in user_positions:
                user_pos = user_positions[user.id]
                for sat in self.satellites:
                    if sat.id in sat_positions:
                        # 检查可见性
                        visibility_key = (user.id, sat.id)
                        if visibility_key in self.user_sat_visibility and self.user_sat_visibility[visibility_key] > 0:
                            current_sat_pos = sat_positions[sat.id]  # 使用不同的变量名避免冲突
                            # 绘制所有可见性连接，包括当前服务卫星的可见性
                            # 如果是当前服务卫星，使用不同的样式
                            if user.current_sat is not None and user.current_sat.id == sat.id:
                                # 当前服务卫星的可见性连接（绿色虚线）
                                link_trace = go.Scatter3d(
                                    x=[user_pos[0], current_sat_pos[0]],
                                    y=[user_pos[1], current_sat_pos[1]],
                                    z=[user_pos[2], current_sat_pos[2]],
                                    mode='lines',
                                    line=dict(color='green', width=4, dash='solid'),
                                    opacity=0.7,
                                    showlegend=False,
                                    name=f"Current Service Visibility"
                                )
                            else:
                                # 其他可见性连接（灰色虚线）
                                link_trace = go.Scatter3d(
                                    x=[user_pos[0], current_sat_pos[0]],
                                    y=[user_pos[1], current_sat_pos[1]],
                                    z=[user_pos[2], current_sat_pos[2]],
                                    mode='lines',
                                    line=dict(color='green', width=4, dash='dash'),
                                    opacity=0.3,
                                    showlegend=False,
                                    name=f"Visibility Link"
                                )
                            visibility_links_traces.append(link_trace)

        # ---------------- 布局 ---------------- #
        layout = go.Layout(
            title=f"Step {step_idx} 卫星与用户三维分布",
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
                aspectmode='data'
            ),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # 合并所有轨迹
        all_traces = [surface, sat_trace, user_trace] + sat_links_traces + user_sat_links_traces + visibility_links_traces
        fig = go.Figure(data=all_traces, layout=layout)

        # ---------------- 保存 HTML ---------------- #
        save_path = os.path.join(save_dir, f"satellite_step_{step_idx}.html")
        fig.write_html(save_path)

        # ---------------- 详细的调试信息 ---------------- #
        print(f"已保存交互式三维图: {save_path}")
        print(f"连接信息: 卫星间连接 {len(sat_links_traces)} 条, 用户-卫星服务连接 {len(user_sat_links_traces)} 条, 可见性连接 {len(visibility_links_traces)} 条")
        
        # 打印详细的连接信息
        print("=== 绘图调试信息 ===")
        print("卫星间连接关系:")
        for sat1_id, connected_sats in self.sat_topology.items():
            print(f"  卫星{sat1_id} -> {connected_sats}")
        
        print("用户-卫星服务连接:")
        for user in self.user_clusters:
            if user.current_sat is not None:
                print(f"  用户{user.id} -> 卫星{user.current_sat.id}")
            else:
                print(f"  用户{user.id} -> 无服务卫星")
        
        print("用户-卫星可见性连接:")
        visible_count = 0
        for user in self.user_clusters:
            for sat in self.satellites:
                visibility_key = (user.id, sat.id)
                if visibility_key in self.user_sat_visibility and self.user_sat_visibility[visibility_key] > 0:
                    distance = self.user_sat_visibility[visibility_key]
                    print(f"  用户{user.id} -> 卫星{sat.id} (距离: {distance:.2f}km)")
                    visible_count += 1
        
        print(f"总计: {visible_count} 条可见性连接")
        
        # 添加更详细的调试信息
        print("=== 详细调试信息 ===")
        print("用户位置信息:")
        for user in self.user_clusters:
            print(f"  用户{user.id}: 经度={user.lon:.2f}, 纬度={user.lat:.2f}")
        
        print("卫星位置信息:")
        for sat in self.satellites:
            pos = sat._satellite_pos(self.current_time)
            print(f"  卫星{sat.id}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
        
        print("用户current_sat详细信息:")
        for user in self.user_clusters:
            if user.current_sat is not None:
                print(f"  用户{user.id}.current_sat.id = {user.current_sat.id}")
                print(f"  用户{user.id}.current_sat 对象地址: {id(user.current_sat)}")
            else:
                print(f"  用户{user.id}.current_sat = None")
        
        print("卫星ID映射:")
        for i, sat in enumerate(self.satellites):
            print(f"  索引{i} -> 卫星ID{sat.id}")
        
        print("===================")
