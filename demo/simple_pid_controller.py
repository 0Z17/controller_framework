import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.dirname(__file__))

from controller_base import ControllerBase, ControllerOutput
from parameter_manager import ParameterManager
from state_interface import StateManager

class SimplePIDController(ControllerBase):
    """简单的PID位置控制器"""
    
    def _initialize_controller(self):
        """初始化PID控制器"""
        # PID积分项
        self._integral_error = np.zeros(3)
        self._last_error = np.zeros(3)
        self._first_run = True
        
        # 获取物理参数
        self._mass = self.get_param('mass')
        self._gravity = self.get_param('gravity')
    
    def _compute_control(self) -> ControllerOutput:
        """计算PID控制输出"""
        output = ControllerOutput()
        
        # 获取当前状态和参考状态
        current_pos = self.get_position()
        ref_pos = self.get_reference_position()
        current_vel = self.get_velocity()
        ref_vel = self.get_reference_velocity()
        
        # 计算位置误差
        pos_error = ref_pos - current_pos
        vel_error = ref_vel - current_vel
        
        # 获取PID参数
        kp = self.get_param('position_kp')
        ki = self.get_param('position_ki')
        kd = self.get_param('position_kd')
        
        # 计算时间步长（从控制器频率获取）
        dt = 1.0 / self._control_frequency
        
        # PID计算
        # 比例项
        p_term = kp * pos_error
        
        # 积分项
        self._integral_error += pos_error * dt
        # 积分限幅
        integral_limit = self.get_param('integral_limit')
        self._integral_error = np.clip(self._integral_error, -integral_limit, integral_limit)
        i_term = ki * self._integral_error
        
        # 微分项（使用速度误差代替位置误差的微分）
        d_term = kd * vel_error
        
        # 总控制量（期望加速度）
        desired_acc = p_term + i_term + d_term
        
        # 转换为推力（考虑重力补偿）
        gravity_compensation = np.array([0, 0, self._mass * self._gravity])
        thrust = self._mass * desired_acc + gravity_compensation
        
        # 限制推力
        max_thrust = self.get_param('max_thrust')
        thrust_magnitude = np.linalg.norm(thrust)
        if thrust_magnitude > max_thrust:
            thrust = thrust * (max_thrust / thrust_magnitude)
        
        # 填充输出
        output.thrust = thrust
        output.torque = np.zeros(3)  # 简化：不考虑姿态控制
        output.is_valid = True
        
        # 调试信息
        output.debug_info = {
            'pos_error': pos_error.tolist(),
            'vel_error': vel_error.tolist(),
            'p_term': p_term.tolist(),
            'i_term': i_term.tolist(),
            'd_term': d_term.tolist(),
            'desired_acc': desired_acc.tolist(),
            'thrust_magnitude': thrust_magnitude
        }
        
        # 更新上次误差
        self._last_error = pos_error.copy()
        self._first_run = False
        
        return output
    
    def reset_integral(self):
        """重置积分项"""
        self._integral_error.fill(0.0)
        self._first_run = True