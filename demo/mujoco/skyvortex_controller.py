#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkyVortex 六旋翼无人机控制器
基于MuJoCo物理仿真的Python控制器

作者: AI Assistant
日期: 2024
"""

import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math
from typing import Tuple, List, Optional

class SkyVortexController:
    """
    SkyVortex 六旋翼无人机控制器类
    
    功能包括:
    - 基本飞行控制 (起飞、降落、悬停)
    - 姿态控制 (俯仰、横滚、偏航)
    - 位置控制 (x, y, z轴移动)
    - 操作臂控制
    - 传感器数据读取
    """

    def __init__(self, model_path: str = "scene.xml"):
        """
        初始化控制器
        
        Args:
            model_path: MuJoCo模型文件路径
        """
        # 加载MuJoCo模型
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # 控制参数
        self.dt = self.model.opt.timestep  # 仿真时间步长
        self.gravity = 9.81
        self.mass = 7.1077  # 无人机质量 (kg)

        # 积分误差存储
        self.integral_errors = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        
        # 添加几何控制器参数
        self.geometric_params = {
            'position': {'kp': 10.0, 'kv': 15, 'ki': 0.2},
            'attitude': {'kr': 15.0, 'kw': 10.5},
            'altitude': {'kp': 5.0, 'ki': 0.2, 'kd': 1.0}
        }

        # 添加旋转误差矩阵存储
        self.e_R = np.zeros((3, 3))
        
        # 目标状态
        self.target_position = np.array([0.0, 0.0, 3.0])  # 目标位置 [x, y, z]
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # 目标位置 [x, y, z]
        self.target_attitude = np.array([0.0, 0.0, 0.0])   # 目标姿态 [roll, pitch, yaw]
        self.target_attitude_rate = np.array([0.0, 0.0, 0.0])   # 目标姿态 [roll, pitch, yaw]
        self.target_operator_angle = 0.0  # 目标操作臂角度
        
        # 推进器映射
        self.thrust_actuators = [
            'thrust0', 'thrust1', 'thrust2', 
            'thrust3', 'thrust4', 'thrust5'
        ]
        
        # 获取执行器ID
        self._get_actuator_ids()
        
        # 获取传感器ID
        self._get_sensor_ids()
        
        print("SkyVortex控制器初始化完成")
        print(f"模型包含 {self.model.nu} 个执行器")
        print(f"模型包含 {self.model.nsensor} 个传感器")
    
    def _get_actuator_ids(self):
        """获取执行器ID"""
        self.actuator_ids = {}
        
        # 推进器ID
        for i, name in enumerate(self.thrust_actuators):
            self.actuator_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        
        # 操作臂执行器ID
        self.actuator_ids['operator_joint'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'operator_joint')
        
        # 力/力矩执行器ID
        for i in range(6):
            name = f'ft{i}'
            self.actuator_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
    
    def _get_sensor_ids(self):
        """获取传感器ID"""
        self.sensor_ids = {}
        sensor_names = ['body_gyro', 'body_acc', 'body_quat', 'body_pos', 'body_vel', 'contact_force']
        
        for name in sensor_names:
            self.sensor_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
    
    def get_state(self) -> dict:
        """
        获取无人机当前状态
        
        Returns:
            包含位置、姿态、速度等信息的字典
        """
        # 获取位置 (IMU传感器)
        pos_sensor_id = self.sensor_ids['body_pos']
        position = self.data.sensordata[self.model.sensor_adr[pos_sensor_id]:self.model.sensor_adr[pos_sensor_id]+3]
        
        # 获取四元数姿态
        quat_sensor_id = self.sensor_ids['body_quat']
        quaternion = self.data.sensordata[self.model.sensor_adr[quat_sensor_id]:self.model.sensor_adr[quat_sensor_id]+4]
        
        # 转换为欧拉角
        euler = self._quat_to_euler(quaternion)
        
        # 转换为旋转矩阵
        rotation_matrix = self._quat_to_rotation_matrix(quaternion)

        # 获取线速度
        vel_sensor_id = self.sensor_ids['body_vel']
        velocity = self.data.sensordata[self.model.sensor_adr[vel_sensor_id]:self.model.sensor_adr[vel_sensor_id]+3]
        
        # 获取角速度
        gyro_sensor_id = self.sensor_ids['body_gyro']
        angular_velocity = self.data.sensordata[self.model.sensor_adr[gyro_sensor_id]:self.model.sensor_adr[gyro_sensor_id]+3]
        
        # 获取加速度
        acc_sensor_id = self.sensor_ids['body_acc']
        acceleration = self.data.sensordata[self.model.sensor_adr[acc_sensor_id]:self.model.sensor_adr[acc_sensor_id]+3]
        
        return {
            'position': position.copy(),
            'attitude': euler,
            'rotation_matrix': rotation_matrix,
            'quaternion': quaternion.copy(),
            'velocity': velocity.copy(),
            'angular_velocity': angular_velocity.copy(),
            'acceleration': acceleration.copy()
        }
    
    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转欧拉角 (roll, pitch, yaw)
        
        Args:
            quat: 四元数 [w, x, y, z]

        Returns:
            欧拉角 [roll, pitch, yaw] (弧度)
        """
        w, x, y, z = quat
        
        # Roll (x轴旋转)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y轴旋转)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z轴旋转)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """
        欧拉角转旋转矩阵
        (roll, pitch, yaw) -> 3x3旋转矩阵
        
        Args:
            euler: [roll, pitch, yaw] 弧度
            
        Returns:
            3x3 旋转矩阵
        """
        roll, pitch, yaw = euler
        
        # 计算每个轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        
        R_y = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        
        R_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵 (Z-Y-X顺序)
        return R_z.dot(R_y.dot(R_x))

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转旋转矩阵
        Args:
            quat: 四元数 [w, x, y, z]  
        Returns:
            3x3 旋转矩阵
        """
        w, x, y, z = quat
        
        # 计算旋转矩阵的各个元素
        R11 = 1 - 2 * (y * y + z * z)
        R12 = 2 * (x * y - w * z)
        R13 = 2 * (x * z + w * y)
        
        R21 = 2 * (x * y + w * z)
        R22 = 1 - 2 * (x * x + z * z)
        R23 = 2 * (y * z - w * x)
        
        R31 = 2 * (x * z - w * y)
        R32 = 2 * (y * z + w * x)
        R33 = 1 - 2 * (x * x + y * y)
        
        # 构造旋转矩阵
        rotation_matrix = np.array([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]
        ])
        
        return rotation_matrix
 
    def _calculate_allocation_matrix(self):
        """
        计算六维力/力矩到六个电机推力的分配矩阵
        
        Returns:
            6x6 分配矩阵
        """
        # 定义电机位置 [x, y, z]（机体坐标系）
        rotor_positions = [
            np.array([0.44207, 0.2778, 0.033856]),
            np.array([0.019547, 0.52175, 0.033856]),
            np.array([-0.46162, 0.24395, 0.033856]),
            np.array([-0.46162, -0.24395, 0.033856]),
            np.array([0.019547, -0.52175, 0.033856]),
            np.array([0.442074, -0.277802, 0.0338557])
        ]
        
        # 重心位置（从XML获取）
        com = np.array([0, -0.002137, 0.004638])
        
        # 计算相对于重心的位置向量
        rotor_positions_relative = [pos - com for pos in rotor_positions]
        
        # 电机推力方向（机体坐标系，单位向量）[7,8](@ref)
        thrust_directions = [
            np.array([-0.249999, 0.433009, 0.866028]),  # 旋翼0
            np.array([0.499998, 0, 0.866027]),          # 旋翼1
            np.array([-0.249999, -0.433009, 0.866028]), # 旋翼2
            np.array([-0.249999, 0.433009, 0.866028]),  # 旋翼3
            np.array([0.499998, 0, 0.866027]),          # 旋翼4
            np.array([-0.25, -0.433013, 0.866025])      # 旋翼5
        ]
        
        # 归一化推力方向（确保单位向量）
        thrust_directions = [v / np.linalg.norm(v) for v in thrust_directions]
        
        # 电机旋转方向（反扭矩符号）[6](@ref)
        rotation_signs = [1, -1, 1, -1, 1, -1]  # 0,2,4逆时针；1,3,5顺时针
        torque_coeff = 0.06  # 扭矩系数（来自XML gear属性）
        
        # 初始化分配矩阵（6行 x 6列）
        # 行：Fx, Fy, Fz, Mx, My, Mz
        # 列：每个电机的贡献
        B = np.zeros((6, 6))
        
        for i in range(6):
            # 获取当前电机的参数
            F = thrust_directions[i]          # 推力方向（单位向量）
            r = rotor_positions_relative[i]   # 相对于重心的位置
            
            # 计算力矩（r × F）[4](@ref)
            M = np.cross(r, F)
            
            # 添加反扭矩（只影响Mz分量）[1](@ref)
            M[2] += rotation_signs[i] * torque_coeff
            
            # 填入分配矩阵（第i列）
            B[:3, i] = F  # 力分量
            B[3:, i] = M  # 力矩分量
        
        return B

    def geometric_attitude_control(self, state: dict) -> np.ndarray:
        """
        基于几何控制理论的姿态控制器
        
        Args:
            state: 当前状态字典
            dt: 时间步长
            
        Returns:
            3维力矩向量 [Mx, My, Mz]
        """
        # 获取当前状态
        R = state['rotation_matrix']  # 当前旋转矩阵
        omega = state['angular_velocity']  # 当前角速度
        # print(f"角速度: ({omega[0]:.2f}, {omega[1]:.2f}, {omega[2]:.2f})")
        # 目标姿态转换为旋转矩阵
        target_rot = self.euler_to_rotation_matrix(self.target_attitude)
        
        # 计算姿态误差 e_R = 0.5*(R_desᵀR - RᵀR_des) (对数映射)
        R_error = 0.5 * (target_rot.T.dot(R) - R.T.dot(target_rot))
        # 提取误差向量的反对称部分 (vee映射)
        e_R_vec = np.array([R_error[2, 1], R_error[0, 2], R_error[1, 0]])
        # print(f"姿态误差: ({e_R_vec[0]:.2f}, {e_R_vec[1]:.2f}, {e_R_vec[2]:.2f})")
        # 计算角速度误差 e_ω = ω - RᵀR_desω_des
        # (假设目标角速度ω_des=0，即稳定飞行)
        e_omega = omega
        
        # 几何控制律计算力矩
        kr = self.geometric_params['attitude']['kr']
        kw = self.geometric_params['attitude']['kw']
        inertia = np.diag([0.08, 0.12, 0.1])  # 无人机惯量
        
        # 力矩计算公式: τ = -kᵣeᵣ - kᵥeᵥ + ω×(Jω)
        torque = -kr * e_R_vec - kw * e_omega + np.cross(omega, inertia.dot(omega))
        print(f"力矩: ({torque[0]:.2f}, {torque[1]:.2f}, {torque[2]:.2f})")
        return torque

    def compute_thrust_commands(self, state: dict) -> np.ndarray:
        """
        计算推进器推力命令
        
        Args:
            state: 当前状态字典
            
        Returns:
            6个推进器的推力命令
        """
        position = state['position']
        
        velocity = state['velocity']
        rotation_matrix = state['rotation_matrix']

        # === 1. 在惯性系中计算误差 ===
        pos_error_inertial = self.target_position - position
        vel_error_inertial = self.target_velocity - velocity
        # print(f"位置误差: ({pos_error_inertial[0]:.2f}, {pos_error_inertial[1]:.2f}, {pos_error_inertial[2]:.2f})")
        # print(f"速度误差: ({vel_error_inertial[0]:.2f}, {vel_error_inertial[1]:.2f}, {vel_error_inertial[2]:.2f})")
        # 位置控制 (PD)
        kp = self.geometric_params['position']['kp']
        kv = self.geometric_params['position']['kv']

        des_acc_inertial = np.array([
            kp * pos_error_inertial[0] + kv * vel_error_inertial[0],
            kp * pos_error_inertial[1] + kv * vel_error_inertial[1],
            kp * pos_error_inertial[2] + kv * vel_error_inertial[2]])
        
        # === 3. 正确重力补偿（在惯性系中）===
        gravity_inertial = np.array([0, 0, self.gravity])  # 注意是正值
        total_force_inertial = self.mass * (des_acc_inertial + gravity_inertial)

        # 几何姿态控制计算扭矩
        torque = self.geometric_attitude_control(state)

        wrench = np.concatenate((total_force_inertial, torque))
        A  = self._calculate_allocation_matrix()
        # 计算推力分配
        thrust_commands = np.linalg.pinv(A) @ wrench
        # 限制推力范围 (0-100N)
        thrust_commands = np.clip(thrust_commands, 0, 100)

        return thrust_commands
    
    def set_target_position(self, x: float, y: float, z: float):
        """
        设置目标位置
        
        Args:
            x, y, z: 目标位置坐标
        """
        self.target_position = np.array([x, y, z])
        print(f"目标位置设置为: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    def set_target_attitude(self, roll: float, pitch: float, yaw: float):
        """
        设置目标姿态
        
        Args:
            roll, pitch, yaw: 目标姿态角 (弧度)
        """
        self.target_attitude = np.array([roll, pitch, yaw])
        print(f"目标姿态设置为: Roll={math.degrees(roll):.1f}°, Pitch={math.degrees(pitch):.1f}°, Yaw={math.degrees(yaw):.1f}°")
    
    def set_operator_angle(self, angle: float):
        """
        设置操作臂角度
        
        Args:
            angle: 目标角度 (弧度)
        """
        self.target_operator_angle = angle
        operator_id = self.actuator_ids['operator_joint']
        self.data.ctrl[operator_id] = angle
        print(f"操作臂角度设置为: {math.degrees(angle):.1f}°")
    
    def takeoff(self, target_height: float):
        """
        起飞到指定高度
        
        Args:
            target_height: 目标高度
        """
        current_pos = self.get_state()['position']
        # self.set_target_position(current_pos[0], current_pos[1], target_height)
        self.set_target_position(0.0, 0.0, target_height)
        print(f"开始起飞到高度 {target_height:.1f}m")
    
    def land(self):
        """
        降落
        """
        current_pos = self.get_state()['position']
        self.set_target_position(current_pos[0], current_pos[1], 0.35)
        print("开始降落")
    
    def hover(self):
        """
        悬停在当前位置
        """
        current_pos = self.get_state()['position']
        self.set_target_position(current_pos[0], current_pos[1], current_pos[2])
        print(f"悬停在位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
    
    def move_to(self, x: float, y: float, z: float):
        """
        移动到指定位置
        
        Args:
            x, y, z: 目标位置
        """

        self.set_target_position(x, y, z)
    
    def update_control(self):
        """
        更新控制命令 (每个仿真步调用)
        """
        # 获取当前状态
        state = self.get_state()
        
        # 计算推力命令
        thrust_commands = self.compute_thrust_commands(state)
        
        # 应用推力命令
        for i, thrust_name in enumerate(self.thrust_actuators):
            actuator_id = self.actuator_ids[thrust_name]
            self.data.ctrl[actuator_id] = thrust_commands[i]
    
    def print_status(self):
        """
        打印当前状态信息
        """
        state = self.get_state()
        pos = state['position']
        att = state['attitude']
        
        print(f"\r位置: ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}) | "
              f"姿态: R={math.degrees(att[0]):6.1f}° P={math.degrees(att[1]):6.1f}° Y={math.degrees(att[2]):6.1f}° | "
              f"目标: ({self.target_position[0]:6.2f}, {self.target_position[1]:6.2f}, {self.target_position[2]:6.2f})", end="")

def main():
    """
    主函数 - 演示控制器使用
    """
    print("=== SkyVortex 控制器演示 ===")
    
    # 创建控制器
    controller = SkyVortexController("scene.xml")
    
        # 启动viewer
    with viewer.launch_passive(controller.model, controller.data) as v:
        print("\n可视化窗口已启动")
        print("控制说明:")
        print("- 仿真会自动执行预设的飞行任务")
        print("- 按 Ctrl+C 可以停止仿真")

        # 仿真参数
        start_time = time.time()
    
        # 任务时间点
        task_times = {
            'takeoff': 3.0,
            'hover': 6.0,
            'move1': 10.0,
            'move2': 15.0,
            'return': 21.0,
            'land': 26.0
        }

        count = 0

        try:
            while v.is_running:
                current_time = time.time() - start_time
                
                # 执行任务
                if abs(current_time - task_times['takeoff']) < controller.dt:
                    controller.takeoff(2.0)
                elif abs(current_time - task_times['hover']) < controller.dt:
                    controller.hover()
                elif abs(current_time - task_times['move1']) < controller.dt:
                    controller.move_to(2.0, 2.0, 3.0)
                elif abs(current_time - task_times['move2']) < controller.dt:
                    controller.move_to(0.0, 0.0, 3.0)
                elif abs(current_time - task_times['land']) < controller.dt:
                    controller.land()
                
                # 更新控制
                controller.update_control()

                #仿真慢放
                count = count + 1
                if count % 1 == 0:
                    # 仿真步进
                    mj.mj_step(controller.model, controller.data)
                
                #更新viewer
                v.sync()

                # 打印状态 (每0.1秒)
                if int(current_time * 10) % 5 == 0:  # 每0.5秒打印一次
                    controller.print_status()
                
                # 简单的延时 (可选)
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n\n仿真被用户中断")
        
        print("\n\n仿真完成")
    
    # 最终状态
    final_state = controller.get_state()
    print(f"\n最终位置: ({final_state['position'][0]:.2f}, {final_state['position'][1]:.2f}, {final_state['position'][2]:.2f})")
    print(f"最终姿态: Roll={math.degrees(final_state['attitude'][0]):.1f}°, Pitch={math.degrees(final_state['attitude'][1]):.1f}°, Yaw={math.degrees(final_state['attitude'][2]):.1f}°")


if __name__ == "__main__":
    main()