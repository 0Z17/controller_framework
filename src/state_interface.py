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
        self.position: np.ndarray = np.zeros(3)  # [x, y, z]
        self.velocity: np.ndarray = np.zeros(3)  # [vx, vy, vz]
        self.acceleration: np.ndarray = np.zeros(3)  # [ax, ay, az]
        self.orientation: np.ndarray = np.zeros(4)  # 四元数 [w, x, y, z]
        self.angular_velocity: np.ndarray = np.zeros(3)  # [wx, wy, wz]
        self.angular_acceleration: np.ndarray = np.zeros(3)  # [dwx, dwy, dwz]
        
        # 参考状态
        self.ref_position: np.ndarray = np.zeros(3)
        self.ref_velocity: np.ndarray = np.zeros(3)
        self.ref_acceleration: np.ndarray = np.zeros(3)
        self.ref_orientation: np.ndarray = np.array([1, 0, 0, 0])  # 单位四元数
        self.ref_angular_velocity: np.ndarray = np.zeros(3)
        self.ref_angular_acceleration: np.ndarray = np.zeros(3)
        
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
                'position': self.position.copy(),
                'velocity': self.velocity.copy(),
                'acceleration': self.acceleration.copy(),
                'orientation': self.orientation.copy(),
                'angular_velocity': self.angular_velocity.copy(),
                'ref_position': self.ref_position.copy(),
                'ref_velocity': self.ref_velocity.copy(),
                'ref_acceleration': self.ref_acceleration.copy(),
                'ref_orientation': self.ref_orientation.copy(),
                'ref_angular_velocity': self.ref_angular_velocity.copy(),
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
            self.position = np.zeros(3)
            self.velocity = np.zeros(3)
            self.acceleration = np.zeros(3)
            self.orientation = np.zeros(4)
            self.angular_velocity = np.zeros(3)
            self.angular_acceleration = np.zeros(3)
            
            # 参考状态
            self.ref_position = np.zeros(3)
            self.ref_velocity = np.zeros(3)
            self.ref_acceleration = np.zeros(3)
            self.ref_orientation = np.array([1, 0, 0, 0])
            self.ref_angular_velocity = np.zeros(3) 
            self.ref_angular_acceleration = np.zeros(3)
            
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
    
    def get_position_error(self) -> np.ndarray:
        """获取位置误差 (ref_position - position)"""
        return self.ref_position - self.position
    
    def get_velocity_error(self) -> np.ndarray:
        """获取速度误差 (ref_velocity - velocity)"""
        return self.ref_velocity - self.velocity
    
    def get_acceleration_error(self) -> np.ndarray:
        """获取加速度误差 (ref_acceleration - acceleration)"""
        return self.ref_acceleration - self.acceleration
    
    def get_angular_velocity_error(self) -> np.ndarray:
        """获取角速度误差 (ref_angular_velocity - angular_velocity)"""
        return self.ref_angular_velocity - self.angular_velocity
    
    def get_angular_acceleration_error(self) -> np.ndarray:
        """获取角加速度误差 (ref_angular_acceleration - angular_acceleration)"""
        return self.ref_angular_acceleration - self.angular_acceleration
    
    def get_orientation_error_quaternion(self) -> np.ndarray:
        """获取四元数形式的姿态误差
        
        Returns:
            四元数误差 [w, x, y, z]，表示从当前姿态到参考姿态的旋转（世界坐标系）
        """
        # 使用 MathUtils 计算四元数误差
        # 误差四元数 = ref_orientation * conjugate(current_orientation)
        current_quat_conj = MathUtils.quaternion_conjugate(self.orientation)
        error_quat = MathUtils.quaternion_multiply(self.ref_orientation, current_quat_conj)
        return error_quat
    
    def get_orientation_error_euler(self, sequence: str = 'xyz') -> np.ndarray:
        """获取欧拉角形式的姿态误差
        
        Args:
            sequence: 欧拉角序列，默认为 'xyz'
            
        Returns:
            欧拉角误差 [roll, pitch, yaw] (弧度)
        """
        # 获取四元数误差
        error_quat = self.get_orientation_error_quaternion()
        
        # 转换为旋转矩阵
        error_rotation_matrix = MathUtils.quaternion_to_rotation_matrix(error_quat)
        
        # 转换为欧拉角
        roll, pitch, yaw = MathUtils.rotation_matrix_to_euler(error_rotation_matrix, sequence)
        
        # 角度归一化到 [-π, π]
        roll = MathUtils.wrap_angle(roll)
        pitch = MathUtils.wrap_angle(pitch)
        yaw = MathUtils.wrap_angle(yaw)
        
        return np.array([roll, pitch, yaw])
    
    def get_position_error_magnitude(self) -> float:
        """获取位置误差的模长"""
        return np.linalg.norm(self.get_position_error())
    
    def get_velocity_error_magnitude(self) -> float:
        """获取速度误差的模长"""
        return np.linalg.norm(self.get_velocity_error())
    
    def get_angular_velocity_error_magnitude(self) -> float:
        """获取角速度误差的模长"""
        return np.linalg.norm(self.get_angular_velocity_error())
    
    def get_orientation_error_magnitude(self) -> float:
        """获取姿态误差的模长（四元数的角度部分）
        
        Returns:
            姿态误差角度 (弧度)
        """
        error_quat = self.get_orientation_error_quaternion()
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
            'position_error': self.get_position_error(),
            'velocity_error': self.get_velocity_error(),
            'acceleration_error': self.get_acceleration_error(),
            'angular_velocity_error': self.get_angular_velocity_error(),
            'angular_acceleration_error': self.get_angular_acceleration_error(),
            'orientation_error_quaternion': self.get_orientation_error_quaternion(),
            'orientation_error_euler': self.get_orientation_error_euler(),
            'position_error_magnitude': self.get_position_error_magnitude(),
            'velocity_error_magnitude': self.get_velocity_error_magnitude(),
            'angular_velocity_error_magnitude': self.get_angular_velocity_error_magnitude(),
            'orientation_error_magnitude': self.get_orientation_error_magnitude()
        }