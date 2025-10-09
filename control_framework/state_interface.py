import numpy as np
from threading import Lock
from typing import Callable, Dict, Any
from .math_utils import MathUtils

class StateManager:
    """统一的状态管理器，所有状态变量都可以直接访问和修改"""
    
    def __init__(self):
        self._lock = Lock()
        self._callbacks: Dict[str, Callable] = {}
        
        # 飞控状态
        self.mode: str = "MANUAL"

        # 无人机状态
        self.pos: np.ndarray = np.zeros(3)  # [x, y, z]
        self.vel: np.ndarray = np.zeros(3)  # [vx, vy, vz]
        self.vel_body: np.ndarray = np.zeros(3)  # [vx, vy, vz]
        self.acc_body: np.ndarray = np.zeros(3)  # [ax, ay, az]
        self.ori: np.ndarray = np.zeros(4)  # 四元数 [w, x, y, z]
        self.ang_vel_body: np.ndarray = np.zeros(3)  # [wx, wy, wz]
        self.ang_acc_body: np.ndarray = np.zeros(3)  # [dwx, dwy, dwz]
        
        # 参考状态
        self.ref_pos: np.ndarray = np.zeros(3)
        self.ref_vel: np.ndarray = np.zeros(3)
        self.ref_vel_body: np.ndarray = np.zeros(3)
        self.ref_acc_body: np.ndarray = np.zeros(3)
        self.ref_ori: np.ndarray = np.array([1, 0, 0, 0])  # 单位四元数
        self.ref_ang_vel_body: np.ndarray = np.zeros(3)
        self.ref_ang_acc_body: np.ndarray = np.zeros(3)
        
        # 力状态
        self.contact_force: np.ndarray = np.zeros(3)
        self.estimated_force: np.ndarray = np.zeros(3)
        
        # 控制输出
        self.thrust: np.ndarray = np.zeros(3)
        self.torque: np.ndarray = np.zeros(3)
        
        # 时间戳
        self.timestamp: float = 0.0
    
    def register_callback(self, event_name: str, callback: Callable):
        """注册状态变化回调函数"""
        with self._lock:
            self._callbacks[event_name] = callback
    
    def trigger_callback(self, event_name: str):
        """手动触发回调函数（可选使用）"""
        if event_name in self._callbacks:
            try:
                self._callbacks[event_name]()
            except Exception as e:
                print(f"回调函数执行错误: {e}")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取所有状态的字典表示"""
        with self._lock:
            return {
                'mode': self.mode,
                'position': self.pos.copy(),
                'velocity': self.vel_body.copy(),
                'acceleration': self.acc_body.copy(),
                'orientation': self.ori.copy(),
                'angular_velocity': self.ang_vel_body.copy(),
                'ref_position': self.ref_pos.copy(),
                'ref_velocity': self.ref_vel_body.copy(),
                'ref_acceleration': self.ref_acc_body.copy(),
                'ref_orientation': self.ref_ori.copy(),
                'ref_angular_velocity': self.ref_ang_vel_body.copy(),
                'contact_force': self.contact_force.copy(),
                'estimated_force': self.estimated_force.copy(),
                'thrust': self.thrust.copy(),
                'torque': self.torque.copy(),
                'timestamp': self.timestamp
            }
    
    def reset_all_states(self):
        """重置所有状态到初始值"""
        with self._lock:
            # 飞控状态
            self.mode = "MANUAL"
            
            # 无人机状态
            self.pos = np.zeros(3)
            self.vel = np.zeros(3)
            self.vel_body = np.zeros(3)
            self.acc_body = np.zeros(3)
            self.ori = np.zeros(4)
            self.ang_vel_body = np.zeros(3)
            self.ang_acc_body = np.zeros(3)
            
            # 参考状态
            self.ref_pos = np.zeros(3)
            self.ref_vel_body = np.zeros(3)
            self.ref_acc_body = np.zeros(3)
            self.ref_ori = np.array([1, 0, 0, 0])
            self.ref_ang_vel_body = np.zeros(3) 
            self.ref_ang_acc_body = np.zeros(3)
            
            # 力状态
            self.contact_force = np.zeros(3)
            self.contact_torque = np.zeros(3)
            self.estimated_force = np.zeros(3)
            self.estimated_torque = np.zeros(3)
            
            # 控制输出
            self.thrust = np.zeros(3)
            self.torque = np.zeros(3)
            
            # 时间戳
            self.timestamp = 0.0

    # ==================== 错误计算接口 ====================
    
    def get_pos_err(self) -> np.ndarray:
        """获取位置误差 (ref_position - position)"""
        return self.ref_pos - self.pos

    def get_vel_err(self) -> np.ndarray:
        """获取速度误差 (ref_velocity - velocity)"""
        return self.ref_vel - self.vel

    def get_vel_body_err(self) -> np.ndarray:
        """获取速度误差 (ref_velocity - velocity)"""
        return self.ref_vel_body - self.vel_body
    
    def get_acc_body_err(self) -> np.ndarray:
        """获取加速度误差 (ref_acceleration - acceleration)"""
        return self.ref_acc_body - self.acc_body
    
    def get_ang_vel_body_err_so3(self) -> np.ndarray:
        """获取角速度误差 (ref_angular_velocity - angular_velocity)"""
        R_ref = MathUtils.quaternion_to_rotation_matrix(self.ref_ori)
        R_curret = MathUtils.quaternion_to_rotation_matrix(self.ori)
        err = R_curret.T @ R_ref @ self.ref_ang_vel_body - self.ang_vel_body
        return err
    
    def get_ang_acc_body_err(self) -> np.ndarray:
        """获取角加速度误差 (ref_angular_acceleration - angular_acceleration)"""
        return self.ref_ang_acc_body - self.ang_acc_body
    
    def get_ori_err_quaternion(self) -> np.ndarray:
        """获取四元数形式的姿态误差
        
        Returns:
            四元数误差 [w, x, y, z]，表示从当前姿态到参考姿态的旋转（机体坐标系）
        """
        # 使用 MathUtils 计算四元数误差
        # 误差四元数 = ref_orientation * conjugate(current_orientation)
        current_quat_conj = MathUtils.quaternion_conjugate(self.ori)
        error_quat = MathUtils.quaternion_multiply(current_quat_conj,self.ref_ori)
        return error_quat
    
    def get_ori_err_euler(self, sequence: str = 'xyz') -> np.ndarray:
        """获取欧拉角形式的姿态误差
        
        Args:
            sequence: 欧拉角序列，默认为 'xyz'
            
        Returns:
            欧拉角误差 [roll, pitch, yaw] (弧度)
        """
        # 获取四元数误差
        error_quat = self.get_ori_err_quaternion()
        
        # 转换为旋转矩阵
        error_rotation_matrix = MathUtils.quaternion_to_rotation_matrix(error_quat)
        
        # 转换为欧拉角
        roll, pitch, yaw = MathUtils.rotation_matrix_to_euler(error_rotation_matrix, sequence)
        
        # 角度归一化到 [-π, π]
        roll = MathUtils.wrap_angle(roll)
        pitch = MathUtils.wrap_angle(pitch)
        yaw = MathUtils.wrap_angle(yaw)
        
        return np.array([roll, pitch, yaw])

    def get_ori_body_err_so3(self) -> np.ndarray:
        """获取SO(3)形式的姿态误差(机体坐标系)
        
        Returns:
            旋转矩阵误差，将当前姿态旋转到参考姿态
        """
        # 计算当前姿态的旋转矩阵
        current_rotation_matrix = MathUtils.quaternion_to_rotation_matrix(self.ori)
        
        # 计算参考姿态的旋转矩阵
        ref_rotation_matrix = MathUtils.quaternion_to_rotation_matrix(self.ref_ori)
        
        # 计算误差旋转矩阵 = 参考旋转矩阵 * 转置(当前旋转矩阵)
        err = 1/2 * MathUtils.vex_operator(
            current_rotation_matrix.T @ ref_rotation_matrix -
            ref_rotation_matrix.T @ current_rotation_matrix
            )
        
        return err
    
    def get_pos_err_magnitude(self) -> float:
        """获取位置误差的模长"""
        return np.linalg.norm(self.get_pos_err())
    
    def get_vel_body_err_magnitude(self) -> float:
        """获取速度误差的模长"""
        return np.linalg.norm(self.get_vel_body_err())
    
    def get_ang_vel_body_err_magnitude(self) -> float:
        """获取角速度误差的模长"""
        return np.linalg.norm(self.get_ang_vel_body_err())
    
    def get_ori_err_magnitude(self) -> float:
        """获取姿态误差的模长（四元数的角度部分）
        
        Returns:
            姿态误差角度 (弧度)
        """
        error_quat = self.get_ori_err_quaternion()
        # 四元数 [w, x, y, z] 对应的旋转角度为 2 * arccos(|w|)
        w = abs(error_quat[0])
        # 确保 w 在有效范围内
        w = min(1.0, max(0.0, w))
        angle = 2.0 * np.arccos(w)
        return angle
    
    def get_all_errors(self) -> Dict[str, Any]:
        """获取所有误差的字典表示
        
        Returns:
            包含所有误差信息的字典
        """
        return {
            'position_error': self.get_pos_err(),
            'velocity_error': self.get_vel_body_err(),
            'acceleration_error': self.get_acc_body_err(),
            'angular_velocity_error': self.get_ang_vel_body_err(),
            'angular_acceleration_error': self.get_ang_acc_body_err(),
            'orientation_error_quaternion': self.get_ori_err_quaternion(),
            'orientation_error_euler': self.get_ori_err_euler(),
            'position_error_magnitude': self.get_pos_err_magnitude(),
            'velocity_error_magnitude': self.get_vel_body_err_magnitude(),
            'angular_velocity_error_magnitude': self.get_ang_vel_body_err_magnitude(),
            'orientation_error_magnitude': self.get_ori_err_magnitude()
        }