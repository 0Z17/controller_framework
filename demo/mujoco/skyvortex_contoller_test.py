"""
SkyVortex 无人机 Mujoco 仿真控制测试脚本

基于 control_framework 实现的几何控制器在 Mujoco 环境中的闭环控制测试
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os
from typing import Dict, Any
import zmq
import json

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from control_framework import ParameterManager, StateManager, MathUtils
from demo.geometry_controller import GeometryController


class SensorWrapper:
    """传感器数据包装器"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # 传感器ID映射
        self.sensor_ids = {
            'gyro': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'body_gyro'),
            'acc': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'body_acc'),
            'quat': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'body_quat'),
            'pos': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'body_pos'),
            'vel': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'body_vel'),
            'contact_force': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'contact_force')
        }
        
    def get_position(self) -> np.ndarray:
        """获取位置 [x, y, z]"""
        sensor_id = self.sensor_ids['pos']
        return self.data.sensordata[self.model.sensor_adr[sensor_id]:self.model.sensor_adr[sensor_id]+3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """获取速度 [vx, vy, vz]"""
        sensor_id = self.sensor_ids['vel']
        return self.data.sensordata[self.model.sensor_adr[sensor_id]:self.model.sensor_adr[sensor_id]+3].copy()
    
    def get_orientation(self) -> np.ndarray:
        """获取四元数 [w, x, y, z]"""
        sensor_id = self.sensor_ids['quat']
        return self.data.sensordata[self.model.sensor_adr[sensor_id]:self.model.sensor_adr[sensor_id]+4].copy()
    
    def get_angular_velocity(self) -> np.ndarray:
        """获取角速度 [wx, wy, wz]"""
        sensor_id = self.sensor_ids['gyro']
        return self.data.sensordata[self.model.sensor_adr[sensor_id]:self.model.sensor_adr[sensor_id]+3].copy()
    
    def get_acceleration(self) -> np.ndarray:
        """获取加速度 [ax, ay, az]"""
        sensor_id = self.sensor_ids['acc']
        return self.data.sensordata[self.model.sensor_adr[sensor_id]:self.model.sensor_adr[sensor_id]+3].copy()
    
    def get_contact_force(self) -> np.ndarray:
        """获取接触力 [fx, fy, fz]"""
        sensor_id = self.sensor_ids['contact_force']
        return self.data.sensordata[self.model.sensor_adr[sensor_id]:self.model.sensor_adr[sensor_id]+3].copy()


class ActuatorWrapper:
    """执行器控制包装器"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        self.thrust_actuators = [
            'thrust0', 'thrust1', 'thrust2', 
            'thrust3', 'thrust4', 'thrust5'
        ]
        # 推进器ID映射（可选，用于分配控制）
        self.thrust_ids = {
            f'thrust{i}': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'thrust{i}')
            for i in range(6)
        }
        
    def set_force_torque(self, force: np.ndarray, torque: np.ndarray):
        """设置力和力矩控制量
        
        Args:
            force: 3D力向量 [fx, fy, fz]
            torque: 3D力矩向量 [tx, ty, tz]
        """

        thrust = self.set_thrust_allocation(force, torque)

        # 设置力控制
        for i, actuator in enumerate(self.thrust_actuators):
            self.data.ctrl[self.thrust_ids[actuator]] = thrust[i]

    
    def set_thrust_allocation(self, force: np.ndarray, torque: np.ndarray) -> np.ndarray:
        """将力和力矩分配到各个推进器（可选实现）"""
        # 这里可以实现推力分配算法
        J = self._calculate_allocation_matrix()
        thrust = np.linalg.inv(J) @ np.hstack([force, torque])

        return thrust



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
        J = np.zeros((6, 6))
        
        for i in range(6):
            # 获取当前电机的参数
            F = thrust_directions[i]          # 推力方向（单位向量）
            r = rotor_positions_relative[i]   # 相对于重心的位置
            
            # 计算力矩（r × F）[4](@ref)
            M = np.cross(r, F)
            
            # 添加反扭矩（只影响Mz分量）[1](@ref)
            M[2] += rotation_signs[i] * torque_coeff
            
            # 填入分配矩阵（第i列）
            J[:3, i] = F  # 力分量
            J[3:, i] = M  # 力矩分量
        
        return J    


class GeometryControllerParameterManager(ParameterManager):
    """几何控制器参数管理类"""
    
    def _initialize_default_parameters(self):
        """初始化默认参数"""
        self._parameters = {
            'K_pos': np.diag([10.0, 10.0, 10.0]),
            'K_vel': np.diag([0.1, 0.1, 0.1]),
            'K_ori': np.diag([1.0, 1.0, 1.0]),
            'K_ang_vel': np.diag([0.1, 0.1, 0.1]),
            'mass': 7.1077,
            'inertia_matrix': np.array([
                [0.1560383, 0.0, 0.0],
                [0.0, 0.1567601, 0.0],
                [0.0, 0.0, 0.290817]
            ]),
            'gravity': 9.81
        }

class SkyVortexRobot:
    """SkyVortex 机器人类"""
    
    def __init__(self, model_path: str):
        # 加载Mujoco模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化传感器和执行器包装器
        self.sensor_wrapper = SensorWrapper(self.model, self.data)
        self.actuator_wrapper = ActuatorWrapper(self.model, self.data)
        
        # 初始化控制框架组件
        self.parameter_manager = GeometryControllerParameterManager()
        self.state_manager = StateManager()
        
        # 加载参数配置
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'param', 'drone_config.yaml')
        self.parameter_manager.load_from_yaml(config_path)
        
        # 初始化几何控制器
        self.controller = GeometryController(
            parameter_manager=self.parameter_manager,
            state_manager=self.state_manager,
            controller_name="SkyVortex_GeometryController"
        )
        # ZMQ订阅器（遥控器轨迹）
        self._zmq_ctx = None
        self._zmq_sock = None
        
        # 控制参数
        self.control_dt = 0.01  # 10ms控制周期
        self.sim_dt = self.model.opt.timestep
        
        # 设置初始位置
        self.reset_robot()
        
    def _setup_controller_parameters(self):
        """设置几何控制器需要的参数映射"""
        # 从配置文件参数映射到控制器参数
        self.parameter_manager.set_parameter('K_pos', np.diag(self.parameter_manager.get_parameter('position_kp')))
        self.parameter_manager.set_parameter('K_vel', np.diag(self.parameter_manager.get_parameter('position_kd')))
        self.parameter_manager.set_parameter('K_ori', np.diag(self.parameter_manager.get_parameter('attitude_kp')))
        self.parameter_manager.set_parameter('K_ang_vel', np.diag(self.parameter_manager.get_parameter('attitude_kd')))
        
        # 物理参数
        self.parameter_manager.set_parameter('mass', self.parameter_manager.get_parameter('mass'))
        self.parameter_manager.set_parameter('inertia_matrix', np.array(self.parameter_manager.get_parameter('inertia_matrix')))
        self.parameter_manager.set_parameter('gravity', self.parameter_manager.get_parameter('gravity'))
    
    def reset_robot(self):
        """重置机器人到初始状态"""
        # 重置仿真状态
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始位置（悬停在0.5米高度）
        self.data.qpos[2] = 0.5  # z位置
        self.data.qpos[3:7] = [1, 0, 0, 0]  # 四元数 [w,x,y,z]
        
        # 前向仿真一步以更新传感器数据
        mujoco.mj_forward(self.model, self.data)
        
        # 重置状态管理器
        self.state_manager.reset_all_states()
        
        # 设置参考状态（悬停）
        self.state_manager.ref_pos = np.array([0.0, 0.0, 2.0])  # 目标位置
        self.state_manager.ref_vel_body = np.zeros(3)
        self.state_manager.ref_acc_body = np.zeros(3)
        self.state_manager.ref_ori = np.array([1, 0, 0, 0])  # 水平姿态
        self.state_manager.ref_ang_vel_body = np.zeros(3)
        self.state_manager.ref_ang_acc_body = np.zeros(3)

    def start_trajectory_subscriber(self, endpoint: str = "tcp://127.0.0.1:5555", topic: str = "traj"):
        self._zmq_ctx = zmq.Context.instance()
        self._zmq_sock = self._zmq_ctx.socket(zmq.SUB)
        self._zmq_sock.connect(endpoint)
        self._zmq_sock.setsockopt_string(zmq.SUBSCRIBE, topic)

    def poll_and_apply_reference(self) -> bool:
        if self._zmq_sock is None:
            return False
        try:
            parts = self._zmq_sock.recv_multipart(flags=zmq.NOBLOCK)
        except Exception:
            return False
        payload = parts[1].decode()
        msg = json.loads(payload)
        schema = msg.get("schema", [])
        values = msg.get("values", [])
        m = {k: float(values[i]) for i, k in enumerate(schema)}
        x = m.get("x", 0.0)
        y = m.get("y", 0.0)
        z = m.get("z", 0.0)
        yaw = m.get("yaw", 0.0)
        theta = m.get("theta", 0.0)
        # 更新参考轨迹
        self.state_manager.set_ref_position(np.array([x, y, z], dtype=float))
        self.state_manager.set_ref_orientation_yaw(yaw)
        # 更新dummy可视化（如果模型包含dummy）
        # 直接设置mocap位置：base_link_dummy, operator_Link_dummy
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link_dummy")
        if body_id >= 0:
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id >= 0:
                # 基体位置与偏航
                R = MathUtils.euler_to_rotation_matrix(0.0, 0.0, yaw)
                q = MathUtils.rotation_matrix_to_quaternion(R)
                self.data.mocap_pos[mocap_id] = np.array([x, y, z], dtype=float)
                self.data.mocap_quat[mocap_id] = q
                # 操作臂（相对机体y旋转）
                op_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "operator_Link_dummy")
                if op_id >= 0:
                    op_mocap = int(self.model.body_mocapid[op_id])
                    if op_mocap >= 0:
                        # 使用固定相对偏移，同 dummy_robot 默认
                        rel = np.array([0.000870575, 0.000171621, 0.0284551], dtype=float)
                        q_base = q
                        v = np.array([0.0, *rel])
                        v_rot = MathUtils.quaternion_multiply(MathUtils.quaternion_multiply(q_base, v), MathUtils.quaternion_conjugate(q_base))
                        p_world = np.array([x, y, z], dtype=float) + v_rot[1:]
                        R_rel = MathUtils.euler_to_rotation_matrix(0.0, float(theta), 0.0)
                        q_rel = MathUtils.rotation_matrix_to_quaternion(R_rel)
                        q_world = MathUtils.quaternion_multiply(q_base, q_rel)
                        q_world = MathUtils.normalize_quaternion(q_world)
                        self.data.mocap_pos[op_mocap] = p_world
                        self.data.mocap_quat[op_mocap] = q_world
                mujoco.mj_forward(self.model, self.data)
        return True

    def close_trajectory_subscriber(self):
        if self._zmq_sock is not None:
            self._zmq_sock.close(0)
            self._zmq_sock = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
    
    def update_state_from_sensors(self):
        """从传感器更新状态管理器"""
        # 更新当前状态
        self.state_manager.pos = self.sensor_wrapper.get_position()
        self.state_manager.vel = self.sensor_wrapper.get_velocity()
        self.state_manager.acc_body = self.sensor_wrapper.get_acceleration()
        self.state_manager.ori = self.sensor_wrapper.get_orientation()
        self.state_manager.ang_vel_body = self.sensor_wrapper.get_angular_velocity()
        self.state_manager.contact_force = self.sensor_wrapper.get_contact_force()
        self.state_manager.timestamp = self.data.time
    
    def control_step(self):
        """执行一次控制步骤"""
        # 1. 从传感器更新状态
        self.update_state_from_sensors()
        
        # 2. 运行控制器计算
        self.controller.step()
        control_output = self.controller.get_output()
        
        # 3. 将控制输出发送到执行器
        if control_output.is_valid:
            self.actuator_wrapper.set_force_torque(
                control_output.force,
                control_output.torque
            )
            
            # 更新状态管理器中的控制输出
            self.state_manager.force = control_output.force.copy()
            self.state_manager.torque = control_output.torque.copy()
        
        return control_output
    
    def get_state_info(self) -> Dict[str, Any]:
        """获取当前状态信息用于显示"""
        return {
            'position': self.state_manager.pos.copy(),
            'velocity': self.state_manager.vel_body.copy(),
            'orientation': self.state_manager.ori.copy(),
            'angular_velocity': self.state_manager.ang_vel_body.copy(),
            'thrust': self.controller.get_output().force.copy(),
            'torque': self.controller.get_output().torque.copy(),
            'ref_position': self.state_manager.ref_pos.copy(),
            'time': self.data.time
        }


def main():
    """主函数 - 运行仿真"""
    # 模型文件路径 z
    model_path = os.path.join(os.path.dirname(__file__), 'scene.xml')
    
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    try:
        # 创建机器人实例
        robot = SkyVortexRobot(model_path)
        robot.start_trajectory_subscriber(endpoint="tcp://127.0.0.1:5555", topic="traj")
        print("SkyVortex 机器人初始化成功")
        
        # 创建可视化窗口
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            print("开始仿真...")
            print("控制目标：悬停在 [0, 0, 1] 位置")
            print("按 Ctrl+C 退出仿真")
            
            last_control_time = 0.0
            step_count = 0
            
            try:
                while viewer.is_running():
                    step_start_time = time.time()
                    
                    # 控制频率控制
                    if robot.data.time - last_control_time >= robot.control_dt:
                        # 先尝试接收遥控参考并更新
                        robot.poll_and_apply_reference()
                        # 执行控制步骤
                        robot.control_step()
                        last_control_time = robot.data.time
                        ori_err_euler = robot.controller._state_manager.get_ori_err_euler()
                        
                        
                        # 每10步打印一次状态信息
                        if step_count % 10 == 0:
                            state_info = robot.get_state_info()
                            print(f"\n=== 步骤 {step_count} (t={state_info['time']:.2f}s) ===")
                            print(f"位置: [{state_info['position'][0]:.3f}, {state_info['position'][1]:.3f}, {state_info['position'][2]:.3f}]")
                            print(f"姿态误差: [{ori_err_euler[0]:.3f}, {ori_err_euler[1]:.3f}, {ori_err_euler[2]:.3f}]")
                            print(f"目标: [{state_info['ref_position'][0]:.3f}, {state_info['ref_position'][1]:.3f}, {state_info['ref_position'][2]:.3f}]")
                            print(f"推力: [{state_info['thrust'][0]:.2f}, {state_info['thrust'][1]:.2f}, {state_info['thrust'][2]:.2f}]")
                            print(f"力矩: [{state_info['torque'][0]:.2f}, {state_info['torque'][1]:.2f}, {state_info['torque'][2]:.2f}]")
                    
                    # 仿真步进
                    mujoco.mj_step(robot.model, robot.data)
                    
                    # 更新可视化
                    viewer.sync()
                    
                    step_count += 1
                    
                    # 控制仿真速度
                    elapsed = time.time() - step_start_time
                    if elapsed < robot.sim_dt:
                        time.sleep(robot.sim_dt - elapsed)
                        
            except KeyboardInterrupt:
                print("\n仿真被用户中断")
                robot.close_trajectory_subscriber()
                
    except Exception as e:
        print(f"仿真过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()