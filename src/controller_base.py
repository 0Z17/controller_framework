import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from threading import Lock
import time

from parameter_manager import ParameterManager
from state_interface import StateManager

class ControllerOutput:
    """控制器输出数据结构"""
    def __init__(self):
        # 基本控制量
        self.thrust: np.ndarray = np.zeros(3)      # 推力向量
        self.torque: np.ndarray = np.zeros(3)      # 力矩向量
        
        # 状态信息
        self.is_valid: bool = True
        self.timestamp: float = 0.0
        
        # 调试信息
        self.debug_info: Dict[str, Any] = {}
        
        # 控制器特定数据（可扩展）
        self.controller_data: Dict[str, Any] = {}
    
    def reset(self):
        """重置输出"""
        self.thrust.fill(0.0)
        self.torque.fill(0.0)
        self.is_valid = True
        self.timestamp = 0.0
        self.debug_info.clear()
        self.controller_data.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'thrust': self.thrust.tolist(),
            'torque': self.torque.tolist(),
            'is_valid': self.is_valid,
            'timestamp': self.timestamp,
            'debug_info': self.debug_info.copy(),
            'controller_data': self.controller_data.copy()
        }

class ControllerBase(ABC):
    """控制器抽象基类
    
    提供统一的控制器接口，整合参数管理和状态管理功能
    """
    
    def __init__(self, 
                 parameter_manager: ParameterManager,
                 state_manager: StateManager,
                 controller_name: str = "BaseController"):
        """
        初始化控制器
        
        Args:
            parameter_manager: 参数管理器实例
            state_manager: 状态管理器实例
            controller_name: 控制器名称
        """
        self._lock = Lock()
        self._parameter_manager = parameter_manager
        self._state_manager = state_manager
        self._controller_name = controller_name
        
        # 控制器状态
        self._is_enabled = True
        self._last_update_time = 0.0
        self._control_frequency = 100.0
        self._dynamic_frequency = False
        
        # 初始化控制输出
        self._output = ControllerOutput()
        
        # 初始化控制器特定参数
        self._initialize_controller()
    
    @abstractmethod
    def _initialize_controller(self):
        """初始化控制器特定设置 - 子类必须实现"""
        pass

    @abstractmethod
    def _compute_control(self) -> ControllerOutput:
        """计算控制量 - 子类必须实现
        
        Returns:
            ControllerOutput: 控制输出结果
        """
        pass

    # ==================== 参数管理接口 ====================
    
    def get_param(self, name: str) -> Any:
        """获取控制器参数（便捷接口）"""
        return self._parameter_manager.get_parameter(name)
    
    def set_param(self, name: str, value: Any):
        """设置控制器参数（便捷接口）"""
        self._parameter_manager.set_parameter(name, value)
    
    def get_all_params(self) -> Dict[str, Any]:
        """获取所有参数"""
        return self._parameter_manager.get_all_parameters()
    
    def save_params(self, file_name: Optional[str] = None) -> str:
        """保存参数到文件"""
        return self._parameter_manager.save_to_yaml(file_name)
    
    def load_params(self, file_name: Optional[str] = None) -> str:
        """从文件加载参数"""
        return self._parameter_manager.load_from_yaml(file_name)
    
    # ==================== 状态管理接口 ====================
    
    @property
    def state(self) -> StateManager:
        """获取状态管理器（便捷访问）"""
        return self._state_manager
    
    def get_pos(self) -> np.ndarray:
        """获取当前位置"""
        return self._state_manager.pos.copy()
    
    def get_vel_body(self) -> np.ndarray:
        """获取当前速度"""
        return self._state_manager.vel_body.copy()

    def get_acc_body(self) -> np.ndarray:
        """获取当前加速度"""
        return self._state_manager.acc_body.copy()
    
    def get_ori_body(self) -> np.ndarray:
        """获取当前姿态（四元数）"""
        return self._state_manager.ori.copy()
    
    def get_ang_vel_body(self) -> np.ndarray:
        """获取当前角速度"""
        return self._state_manager.ang_vel_body.copy()

    def get_ang_acc_body(self) -> np.ndarray:
        """获取当前角加速度"""
        return self._state_manager.ang_acc_body.copy()
    
    def get_ref_pos(self) -> np.ndarray:
        """获取参考位置"""
        return self._state_manager.ref_pos.copy()
    
    def get_ref_vel_body(self) -> np.ndarray:
        """获取参考速度"""
        return self._state_manager.ref_vel_body.copy()
    
    def get_ref_acc_body(self) -> np.ndarray:
        """获取参考加速度"""
        return self._state_manager.ref_acc_body.copy()
    
    def get_ref_ori(self) -> np.ndarray:
        """获取参考姿态"""
        return self._state_manager.ref_ori.copy()
    
    def get_ref_ang_vel_body(self) -> np.ndarray:
        """获取参考角速度"""
        return self._state_manager.ref_ang_vel_body.copy()

    def get_ref_ang_acc_body(self) -> np.ndarray:
        """获取参考角加速度"""
        return self._state_manager.ref_ang_acc_body.copy()
    
    def get_timestamp(self) -> float:
        """获取当前时间戳"""
        return self._state_manager.timestamp
    
    # ==================== 控制器核心接口 ====================
    
    def step(self) -> ControllerOutput:
        """执行一次控制计算
        
        Args:
            dt: 时间步长（如果为None则自动计算）
            
        Returns:
            ControllerOutput: 控制输出结果
        """
        with self._lock:
            current_time = time.time()
            
            # 计算时间步长
            if self._dynamic_frequency:
                dt = current_time - self._last_update_time
                self._control_frequency = 1.0 / dt
            
            # 检查控制器是否启用
            if not self._is_enabled:
                self._output.reset()
                self._output.is_valid = False
                self._output.debug_info['error'] = "控制器未启用"
                return self._output
            
            try:
                # 执行控制计算
                self._output = self._compute_control()
                self._output.timestamp = current_time
                
                self._state_manager.timestamp = current_time
                
            except Exception as e:
                # 控制计算出错时的处理
                self._output.reset()
                self._output.is_valid = False
                self._output.debug_info['error'] = str(e)
                print(f"控制器 {self._controller_name} 计算错误: {e}")
            
            self._last_update_time = current_time
            return self._output
    
    def get_output(self) -> ControllerOutput:
        """获取最新的控制输出"""
        with self._lock:
            return self._output
    
    def get_output_dict(self) -> Dict[str, Any]:
        """获取控制输出的字典格式"""
        return self._output.to_dict()
    
    def set_control_frequency(self, frequency: float):
        """设置控制频率"""
        with self._lock:
            self._control_frequency = frequency
            self._dynamic_frequency = False

    def set_dt(self, dt: float):
        """设置时间步长"""
        with self._lock:
            self._control_frequency = 1.0 / dt

    def set_dynamic_frequency(self, enable: bool = True):
        """设置是否动态调整频率"""
        with self._lock:
            self._dynamic_frequency = enable

    # ==================== 控制器管理接口 ====================
    
    def enable(self):
        """启用控制器"""
        with self._lock:
            self._is_enabled = True
            print(f"控制器 {self._controller_name} 已启用")
    
    def disable(self):
        """禁用控制器"""
        with self._lock:
            self._is_enabled = False
            self._output.reset()
            self._output.is_valid = False
            print(f"控制器 {self._controller_name} 已禁用")
    
    def is_enabled(self) -> bool:
        """检查控制器是否启用"""
        return self._is_enabled
    
    def reset(self):
        """重置控制器状态"""
        with self._lock:
            self._output.reset()
            self._last_update_time = 0.0
            self._control_frequency = 100.0
            print(f"控制器 {self._controller_name} 已重置")
    
    def get_controller_info(self) -> Dict[str, Any]:
        """获取控制器信息"""
        return {
            'name': self._controller_name,
            'enabled': self._is_enabled,
            'control_frequency': self._control_frequency,
            'last_update_time': self._last_update_time,
            'output_valid': self._output.is_valid,
            'is_dynamic_frequency': self._dynamic_frequency
        }
    
    