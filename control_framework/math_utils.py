"""
数学工具模块
实现四元数转换、旋转矩阵转换等数学运算工具
使用numpy和scipy标准库，避免重新实现已有算法
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple
import warnings


class MathUtils:
    """数学工具类 - 提供通用的数学运算方法"""
    
    @staticmethod
    def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵
        
        Args:
            quaternion: 四元数 [w, x, y, z] (与ROS顺序统一，且已经归一化)
            
        Returns:
            3x3旋转矩阵
        """
        if len(quaternion) != 4:
            raise ValueError("Quaternion must have 4 elements")
        
        quat_scipy = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        
        # 使用scipy转换
        rotation = R.from_quat(quat_scipy)
        return rotation.as_matrix()
    
    @staticmethod
    def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
        """旋转矩阵转四元数
        
        Args:
            rotation_matrix: 3x3旋转矩阵
            
        Returns:
            四元数 [w, x, y, z]
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        
        # 使用scipy转换
        rotation = R.from_matrix(rotation_matrix)
        quat_scipy = rotation.as_quat()  # [x, y, z, w]
        
        # 转换为 [w, x, y, z] 格式
        return np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
    
    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """四元数乘法
        
        Args:
            q1, q2: 四元数 [w, x, y, z]
            
        Returns:
            四元数乘积 [w, x, y, z]
        """
        if len(q1) != 4 or len(q2) != 4:
            raise ValueError("Quaternions must have 4 elements")
        
        # 直接使用四元数乘法公式
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        # 四元数乘法公式: q1 * q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def skew_symmetric_matrix(vector: np.ndarray) -> np.ndarray:
        """向量的反对称矩阵 (叉积矩阵)
        
        Args:
            vector: 3D向量
            
        Returns:
            3x3反对称矩阵
        """
        if len(vector) != 3:
            raise ValueError("Vector must be 3D")
        
        v = vector.flatten()
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    @staticmethod
    def vex_operator(matrix: np.ndarray) -> np.ndarray:
        """反对称矩阵的vex算子 (提取向量)
        
        Args:
            matrix: 3x3反对称矩阵
            
        Returns:
            3D向量
        """
        if matrix.shape != (3, 3):
            raise ValueError("Matrix must be 3x3")
        
        # 检查是否为反对称矩阵
        if not np.allclose(matrix, -matrix.T, atol=1e-6):
            warnings.warn("Matrix is not skew-symmetric")
        
        return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])
    
    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float, 
                                sequence: str = 'xyz') -> np.ndarray:
        """欧拉角转旋转矩阵
        
        Args:
            roll, pitch, yaw: 欧拉角 (弧度)
            sequence: 旋转序列
            
        Returns:
            3x3旋转矩阵
        """
        rotation = R.from_euler(sequence, [roll, pitch, yaw])
        return rotation.as_matrix()
    
    @staticmethod
    def rotation_matrix_to_euler(rotation_matrix: np.ndarray, 
                                sequence: str = 'xyz') -> Tuple[float, float, float]:
        """旋转矩阵转欧拉角
        
        Args:
            rotation_matrix: 3x3旋转矩阵
            sequence: 旋转序列
            
        Returns:
            (roll, pitch, yaw) 欧拉角元组
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler(sequence)
        return tuple(euler_angles)
    
    @staticmethod
    def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        """归一化四元数
        
        Args:
            quaternion: 四元数 [w, x, y, z]
            
        Returns:
            归一化的四元数
        """
        if len(quaternion) != 4:
            raise ValueError("Quaternion must have 4 elements")
        
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("Cannot normalize zero quaternion")
        
        return quaternion / norm
    
    @staticmethod
    def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
        """四元数共轭
        
        Args:
            quaternion: 四元数 [w, x, y, z]
            
        Returns:
            共轭四元数 [w, -x, -y, -z]
        """
        if len(quaternion) != 4:
            raise ValueError("Quaternion must have 4 elements")
        
        return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
    
    @staticmethod
    def rotate_vector_by_quaternion(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """使用四元数旋转向量
        
        Args:
            vector: 3D向量
            quaternion: 四元数 [w, x, y, z]
            
        Returns:
            旋转后的向量
        """
        if len(vector) != 3:
            raise ValueError("Vector must be 3D")
        if len(quaternion) != 4:
            raise ValueError("Quaternion must have 4 elements")
        
        # 使用旋转矩阵方法
        rotation_matrix = MathUtils.quaternion_to_rotation_matrix(quaternion)
        return rotation_matrix @ vector
    
    @staticmethod
    def clamp_vector(vector: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """限制向量的范围
        
        Args:
            vector: 输入向量
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            限制后的向量
        """
        return np.clip(vector, min_val, max_val)
    
    @staticmethod
    def clamp_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        """限制向量的模长
        
        Args:
            vector: 输入向量
            max_norm: 最大模长
            
        Returns:
            限制模长后的向量
        """
        norm = np.linalg.norm(vector)
        if norm > max_norm:
            return vector * (max_norm / norm)
        return vector
    
    @staticmethod
    def low_pass_filter(current_value: float, previous_value: float, 
                       alpha: float) -> float:
        """一阶低通滤波器
        
        Args:
            current_value: 当前值
            previous_value: 前一个值
            alpha: 滤波系数 (0-1)
            
        Returns:
            滤波后的值
        """
        return alpha * current_value + (1 - alpha) * previous_value
    
    @staticmethod
    def wrap_angle(angle: float) -> float:
        """将角度限制在 [-π, π] 范围内
        
        Args:
            angle: 输入角度 (弧度)
            
        Returns:
            限制后的角度
        """
        return np.arctan2(np.sin(angle), np.cos(angle))
