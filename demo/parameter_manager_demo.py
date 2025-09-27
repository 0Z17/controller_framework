import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parameter_manager import ParameterManager

class DroneControllerParameters(ParameterManager):
    """无人机控制器参数管理器的具体实现"""
    
    def _initialize_default_parameters(self):
        """初始化无人机控制器的默认参数"""
        # PID控制器参数
        self._parameters.update({
            # 位置控制PID参数
            'position_kp': np.array([1.0, 1.0, 1.0]),
            'position_ki': np.array([0.1, 0.1, 0.1]),
            'position_kd': np.array([0.5, 0.5, 0.5]),
            
            # 姿态控制PID参数
            'attitude_kp': np.array([2.0, 2.0, 1.0]),
            'attitude_ki': np.array([0.0, 0.0, 0.0]),
            'attitude_kd': np.array([0.1, 0.1, 0.05]),
            
            # 角速度控制PID参数
            'angular_rate_kp': np.array([0.5, 0.5, 0.2]),
            'angular_rate_ki': np.array([0.1, 0.1, 0.05]),
            'angular_rate_kd': np.array([0.01, 0.01, 0.005]),
        })
        
        # 物理参数
        self._parameters.update({
            'mass': 1.5,  # 无人机质量 (kg)
            'gravity': 9.81,  # 重力加速度 (m/s^2)
            'inertia_matrix': np.array([[0.1, 0.0, 0.0],
                                       [0.0, 0.1, 0.0],
                                       [0.0, 0.0, 0.2]]),  # 惯性矩阵
        })
        
        # 限制参数
        self._parameters.update({
            'max_velocity': np.array([5.0, 5.0, 3.0]),  # 最大速度 (m/s)
            'max_acceleration': np.array([3.0, 3.0, 2.0]),  # 最大加速度 (m/s^2)
            'max_angular_velocity': np.array([2.0, 2.0, 1.0]),  # 最大角速度 (rad/s)
            'max_thrust': 30.0,  # 最大推力 (N)
            'max_torque': np.array([1.0, 1.0, 0.5]),  # 最大力矩 (N·m)
        })
        
        # 控制器配置
        self._parameters.update({
            'control_frequency': 100.0,  # 控制频率 (Hz)
            'enable_integral_windup_protection': True,
            'integral_limit': np.array([1.0, 1.0, 1.0]),  # 积分限幅
            'deadzone_threshold': 0.01,  # 死区阈值
        })
        
        # 滤波器参数
        self._parameters.update({
            'lowpass_filter_cutoff': 10.0,  # 低通滤波器截止频率 (Hz)
            'enable_derivative_filter': True,
            'derivative_filter_alpha': 0.1,  # 微分滤波器系数
        })

        self._config_dir_path = os.path.join(os.path.dirname(__file__), '..', 'param')
# 使用示例和测试代码
if __name__ == "__main__":
    # 创建参数管理器实例
    param_manager = DroneControllerParameters()
    
    # 显示所有参数
    print("默认参数:")
    for name in param_manager.list_parameters():
        info = param_manager.get_parameter_info(name)
        print(f"  {name}: {info['value']}")
    
    # 测试参数引用
    print("\n测试参数引用:")
    kp_array = param_manager.get_parameter('position_kp')
    print(f"原始 position_kp: {kp_array}")
    
    # 直接修改数组（因为返回的是引用）
    kp_array[0] = 2.0
    print(f"修改后 position_kp: {param_manager.get_parameter('position_kp')}")
    
    # 使用set_parameter方法（推荐方式）
    param_manager.set_parameter('position_kp', np.array([3.0, 3.0, 3.0]))
    print(f"set_parameter后 position_kp: {param_manager.get_parameter('position_kp')}")
    
    # # 保存到YAML文件
    # print("\n保存到YAML文件...")
    # param_manager.save_to_yaml("drone_config_test.yaml")
    
    # 修改一些参数
    param_manager.set_parameter('mass', 2.0)
    param_manager.set_parameter('control_frequency', 200.0)
    
    # 从YAML文件重新加载
    print("\n从YAML文件重新加载...")
    param_manager.load_from_yaml("drone_config.yaml")
    print(f"重新加载后的质量: {param_manager.get_parameter('mass')}")
    print(f"重新加载后的控制频率: {param_manager.get_parameter('control_frequency')}")