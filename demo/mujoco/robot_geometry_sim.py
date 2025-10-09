#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot + GeometryController Closed-loop Simulation (MuJoCo)
"""
import os
import sys
import time
import math
import numpy as np
import mujoco as mj
import mujoco.viewer as viewer

# 让脚本能找到 src 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from state_interface import StateManager
from parameter_manager import ParameterManager
from geometry_controller import GeometryController
from math_utils import MathUtils

# 参数管理：提供默认值，并兼容 geometry_controller 调用的 get_param
class DroneParameterManager(ParameterManager):
    def _initialize_default_parameters(self):
        # 控制增益（可在 param/drone_config.yaml 中覆盖）
        self._parameters['K_pos'] = np.diag([10.0, 10.0, 10.0])
        self._parameters['K_vel'] = np.diag([15.0, 15.0, 15.0])
        self._parameters['K_ori'] = np.diag([15.0, 15.0, 15.0])
        self._parameters['K_ang_vel'] = np.diag([10.5, 10.5, 10.5])
        # 物理参数（来自 skyvortex.xml 的 base_link）
        self._parameters['mass'] = 7.1077
        self._parameters['inertia_matrix'] = np.diag([0.1560383, 0.1567601, 0.290817])
        self._parameters['gravity'] = 9.81

    # 兼容几何控制器的接口命名
    def get_param(self, name: str):
        return self.get_parameter(name)

class RobotModule:
    def __init__(self, model_path: str = "scene.xml"):
        # 加载 MuJoCo 模型
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.dt = self.model.opt.timestep

        # 状态与控制器
        self.state_manager = StateManager()
        self.param_manager = DroneParameterManager()
        # 参数目录与加载（若存在同名 yaml 则覆盖默认值）
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'param'))
        self.param_manager.set_config_dir_path(config_dir)
        try:
            self.param_manager.load_from_yaml('drone_config.yaml')
        except Exception:
            pass
        self.controller = GeometryController(self.param_manager, self.state_manager, 'GeometryController')

        # 目标状态
        self.target_position = np.array([0.0, 0.0, 2.0])
        self.target_attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.target_attitude_rate = np.zeros(3)

        # 执行器 & 传感器 ID
        self._init_ids()

    def _init_ids(self):
        # 推进器名称
        self.thrust_names = ['thrust0', 'thrust1', 'thrust2', 'thrust3', 'thrust4', 'thrust5']
        self.actuator_ids = {name: mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name) for name in self.thrust_names}
        # 操作臂（可选）
        self.actuator_ids['operator_joint'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'operator_joint')
        # 传感器
        sensor_names = ['body_gyro', 'body_acc', 'body_quat', 'body_pos', 'body_vel']
        self.sensor_ids = {name: mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name) for name in sensor_names}

    def get_state(self) -> dict:
        # 位置
        sid_pos = self.sensor_ids['body_pos']
        pos = self.data.sensordata[self.model.sensor_adr[sid_pos]:self.model.sensor_adr[sid_pos]+3]
        # 姿态四元数 [w, x, y, z]
        sid_quat = self.sensor_ids['body_quat']
        quat = self.data.sensordata[self.model.sensor_adr[sid_quat]:self.model.sensor_adr[sid_quat]+4]
        # 旋转矩阵
        R = self._quat_to_rotation_matrix(quat)
        # 线速度（世界系）
        sid_vel = self.sensor_ids['body_vel']
        vel = self.data.sensordata[self.model.sensor_adr[sid_vel]:self.model.sensor_adr[sid_vel]+3]
        # 角速度（机体系）
        sid_gyro = self.sensor_ids['body_gyro']
        omega = self.data.sensordata[self.model.sensor_adr[sid_gyro]:self.model.sensor_adr[sid_gyro]+3]
        # 加速度（世界系）
        sid_acc = self.sensor_ids['body_acc']
        acc = self.data.sensordata[self.model.sensor_adr[sid_acc]:self.model.sensor_adr[sid_acc]+3]
        return {
            'position': pos.copy(),
            'quaternion': quat.copy(),
            'rotation_matrix': R,
            'velocity': vel.copy(),
            'angular_velocity': omega.copy(),
            'acceleration': acc.copy(),
        }

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        R11 = 1 - 2 * (y * y + z * z)
        R12 = 2 * (x * y - w * z)
        R13 = 2 * (x * z + w * y)
        R21 = 2 * (x * y + w * z)
        R22 = 1 - 2 * (x * x + z * z)
        R23 = 2 * (y * z - w * x)
        R31 = 2 * (x * z - w * y)
        R32 = 2 * (y * z + w * x)
        R33 = 1 - 2 * (x * x + y * y)
        return np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])

    def _update_state_manager(self, s: dict):
        R = s['rotation_matrix']
        # 世界系 -> 机体系 的线速度/加速度
        self.state_manager.pos = s['position'].copy()
        self.state_manager.ori = s['quaternion'].copy()
        self.state_manager.vel_body = R.T @ s['velocity']
        self.state_manager.acc_body = R.T @ s['acceleration']
        self.state_manager.ang_vel_body = s['angular_velocity'].copy()
        # 参考状态（由目标生成）
        self.state_manager.ref_pos = self.target_position.copy()
        self.state_manager.ref_vel_body = np.zeros(3)
        self.state_manager.ref_acc_body = np.zeros(3)
        R_des = self._euler_to_rotation_matrix(self.target_attitude)
        self.state_manager.ref_ori = MathUtils.rotation_matrix_to_quaternion(R_des)
        self.state_manager.ref_ang_vel_body = self.target_attitude_rate.copy()
        self.state_manager.ref_ang_acc_body = np.zeros(3)

    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = euler
        R_x = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll),math.cos(roll)]])
        R_y = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
        R_z = np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
        return R_z @ (R_y @ R_x)

    def _calculate_allocation_matrix(self) -> np.ndarray:
        # rotor positions (body) from skyvortex.xml, relative to COM
        rotor_positions = [
            np.array([0.44207, 0.27780, 0.033856]),
            np.array([0.019547, 0.52175, 0.033856]),
            np.array([-0.46162, 0.24395, 0.033856]),
            np.array([-0.46162,-0.24395, 0.033856]),
            np.array([0.019547,-0.52175, 0.033856]),
            np.array([0.442074,-0.277802,0.0338557]),
        ]
        com = np.array([0.0, -0.002137, 0.004638])
        rotor_positions = [p - com for p in rotor_positions]
        thrust_dirs = [
            np.array([-0.249999,  0.433009, 0.866028]),
            np.array([ 0.499998,  0.000000, 0.866027]),
            np.array([-0.249999, -0.433009, 0.866028]),
            np.array([-0.249999,  0.433009, 0.866028]),
            np.array([ 0.499998,  0.000000, 0.866027]),
            np.array([-0.250000, -0.433013, 0.866025]),
        ]
        thrust_dirs = [v / np.linalg.norm(v) for v in thrust_dirs]
        rotation_signs = [1,-1,1,-1,1,-1]
        torque_coeff = 0.06
        A = np.zeros((6,6))
        for i in range(6):
            F = thrust_dirs[i]
            r = rotor_positions[i]
            M = np.cross(r, F)
            M[2] += rotation_signs[i] * torque_coeff
            A[:3, i] = F
            A[3:, i] = M
        return A

    def compute_thrust_commands(self, s: dict) -> np.ndarray:
        # 更新状态到几何控制器
        self._update_state_manager(s)
        out = self.controller.step()
        force_body = out.thrust
        torque_body = out.torque
        wrench_body = np.concatenate((force_body, torque_body))
        # 分配到 6 个推进器
        A = self._calculate_allocation_matrix()
        thrust = np.linalg.pinv(A) @ wrench_body
        thrust = np.clip(thrust, 0.0, 100.0)
        return thrust

    def apply_thrust(self, thrust: np.ndarray):
        for i, name in enumerate(self.thrust_names):
            self.data.ctrl[self.actuator_ids[name]] = thrust[i]

    # 目标接口
    def set_target_position(self, x: float, y: float, z: float):
        self.target_position = np.array([x, y, z])
    def set_target_attitude(self, roll: float, pitch: float, yaw: float):
        self.target_attitude = np.array([roll, pitch, yaw])
    def set_operator_angle(self, angle: float):
        aid = self.actuator_ids['operator_joint']
        self.data.ctrl[aid] = angle


def main():
    robot = RobotModule("scene.xml")
    # 示例任务
    waypoints = [
        (0.0, 0.0, 1.5, 0.0),
        (0.0, 0.0, 2.0, 3.0),
        (1.0, 0.5, 2.5, 6.0),
        (0.0, 0.0, 2.0, 10.0),
        (0.0, 0.0, 0.4, 14.0),
    ]
    with viewer.launch_passive(robot.model, robot.data) as v:
        start = time.time()
        wp_idx = 0
        print("Viewer started. Running closed-loop control...")
        try:
            while v.is_running:
                t = time.time() - start
                # 切换目标
                if wp_idx < len(waypoints) and t > waypoints[wp_idx][3]:
                    x, y, z, _ = waypoints[wp_idx]
                    robot.set_target_position(x, y, z)
                    wp_idx += 1
                # 读取状态，计算控制并应用
                s = robot.get_state()
                thrust = robot.compute_thrust_commands(s)
                robot.apply_thrust(thrust)
                # 仿真步进与 viewer 同步
                mj.mj_step(robot.model, robot.data)
                v.sync()
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("Interrupted by user.")
        print("Done.")

if __name__ == "__main__":
    main()