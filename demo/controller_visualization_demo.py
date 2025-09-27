import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parameter_manager import ParameterManager
from state_interface import StateManager
from simple_pid_controller import SimplePIDController

class DroneControllerParameters(ParameterManager):
    """无人机控制器参数管理器"""
    
    def _initialize_default_parameters(self):
        """初始化默认参数"""
        # PID控制器参数
        self._parameters.update({
            # 位置控制PID参数
            'position_kp': np.array([2.0, 2.0, 3.0]),
            'position_ki': np.array([0.1, 0.1, 0.2]),
            'position_kd': np.array([1.0, 1.0, 1.5]),
        })
        
        # 物理参数
        self._parameters.update({
            'mass': 1.5,  # 无人机质量 (kg)
            'gravity': 9.81,  # 重力加速度 (m/s^2)
        })
        
        # 限制参数
        self._parameters.update({
            'max_velocity': np.array([5.0, 5.0, 3.0]),
            'max_acceleration': np.array([3.0, 3.0, 2.0]),
            'max_thrust': 30.0,
            'integral_limit': np.array([2.0, 2.0, 2.0]),
        })

class DroneSimulator:
    """简单的无人机动力学仿真器"""
    
    def __init__(self, mass: float = 1.5, gravity: float = 9.81):
        self.mass = mass
        self.gravity = gravity
        
        # 状态变量
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        
        # 添加一些简单的阻尼
        self.damping = 0.1
    
    def update(self, thrust: np.ndarray, dt: float):
        """更新无人机状态"""
        # 计算总力（推力 - 重力 - 阻尼）
        gravity_force = np.array([0, 0, -self.mass * self.gravity])
        damping_force = -self.damping * self.velocity
        total_force = thrust + gravity_force + damping_force
        
        # 计算加速度
        self.acceleration = total_force / self.mass
        
        # 更新速度和位置（欧拉积分）
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

def generate_reference_trajectory(t: float) -> Tuple[np.ndarray, np.ndarray]:
    """生成参考轨迹"""
    # 圆形轨迹
    radius = 2.0
    omega = 0.5  # 角频率
    height = 3.0
    
    # 位置
    ref_pos = np.array([
        radius * np.cos(omega * t),
        radius * np.sin(omega * t),
        height + 0.5 * np.sin(2 * omega * t)
    ])
    
    # 速度
    ref_vel = np.array([
        -radius * omega * np.sin(omega * t),
        radius * omega * np.cos(omega * t),
        0.5 * 2 * omega * np.cos(2 * omega * t)
    ])
    
    return ref_pos, ref_vel

def run_simulation():
    """运行仿真并可视化结果"""
    print("开始控制器仿真...")
    
    # 创建参数管理器和状态管理器
    param_manager = DroneControllerParameters()
    state_manager = StateManager()
    
    # 创建控制器
    controller = SimplePIDController(param_manager, state_manager, "PID_Demo")
    
    # 创建仿真器
    simulator = DroneSimulator(
        mass=param_manager.get_parameter('mass'),
        gravity=param_manager.get_parameter('gravity')
    )
    
    # 仿真参数
    dt = 0.01  # 时间步长
    total_time = 20.0  # 总仿真时间
    steps = int(total_time / dt)
    
    # 数据记录
    time_data = []
    ref_pos_data = []
    actual_pos_data = []
    ref_vel_data = []
    actual_vel_data = []
    thrust_data = []
    error_data = []
    
    # 初始化状态
    simulator.position = np.array([0.0, 0.0, 0.0])
    simulator.velocity = np.array([0.0, 0.0, 0.0])
    
    print(f"仿真参数: dt={dt}s, 总时间={total_time}s, 步数={steps}")
    
    # 仿真循环
    for i in range(steps):
        current_time = i * dt
        
        # 生成参考轨迹
        ref_pos, ref_vel = generate_reference_trajectory(current_time)
        
        # 更新状态管理器
        state_manager.position[:] = simulator.position
        state_manager.velocity[:] = simulator.velocity
        state_manager.ref_position[:] = ref_pos
        state_manager.ref_velocity[:] = ref_vel
        state_manager.timestamp = current_time
        
        # 执行控制
        output = controller.step()
        
        # 更新仿真器
        simulator.update(output.thrust, dt)
        
        # 记录数据（每10步记录一次以减少数据量）
        if i % 10 == 0:
            time_data.append(current_time)
            ref_pos_data.append(ref_pos.copy())
            actual_pos_data.append(simulator.position.copy())
            ref_vel_data.append(ref_vel.copy())
            actual_vel_data.append(simulator.velocity.copy())
            thrust_data.append(output.thrust.copy())
            error_data.append(ref_pos - simulator.position)
        
        # 进度显示
        if i % (steps // 10) == 0:
            progress = (i / steps) * 100
            print(f"仿真进度: {progress:.1f}%")
    
    print("仿真完成，开始绘制结果...")
    
    # 转换为numpy数组便于绘图
    time_data = np.array(time_data)
    ref_pos_data = np.array(ref_pos_data)
    actual_pos_data = np.array(actual_pos_data)
    ref_vel_data = np.array(ref_vel_data)
    actual_vel_data = np.array(actual_vel_data)
    thrust_data = np.array(thrust_data)
    error_data = np.array(error_data)
    
    # 创建图形
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 3D轨迹图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(ref_pos_data[:, 0], ref_pos_data[:, 1], ref_pos_data[:, 2], 
             'r--', linewidth=2, label='期望轨迹')
    ax1.plot(actual_pos_data[:, 0], actual_pos_data[:, 1], actual_pos_data[:, 2], 
             'b-', linewidth=2, label='实际轨迹')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D轨迹跟踪')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 位置跟踪（X-Y平面）
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(ref_pos_data[:, 0], ref_pos_data[:, 1], 'r--', linewidth=2, label='期望轨迹')
    ax2.plot(actual_pos_data[:, 0], actual_pos_data[:, 1], 'b-', linewidth=2, label='实际轨迹')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y平面轨迹')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. 位置误差
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(time_data, error_data[:, 0], 'r-', label='X误差')
    ax3.plot(time_data, error_data[:, 1], 'g-', label='Y误差')
    ax3.plot(time_data, error_data[:, 2], 'b-', label='Z误差')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('位置误差 (m)')
    ax3.set_title('位置跟踪误差')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 位置对比（时间序列）
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(time_data, ref_pos_data[:, 2], 'r--', linewidth=2, label='期望高度')
    ax4.plot(time_data, actual_pos_data[:, 2], 'b-', linewidth=2, label='实际高度')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('高度 (m)')
    ax4.set_title('高度跟踪')
    ax4.legend()
    ax4.grid(True)
    
    # 5. 控制输入
    ax5 = fig.add_subplot(2, 3, 5)
    thrust_magnitude = np.linalg.norm(thrust_data, axis=1)
    ax5.plot(time_data, thrust_magnitude, 'k-', linewidth=2, label='推力大小')
    ax5.axhline(y=param_manager.get_parameter('max_thrust'), color='r', 
                linestyle='--', label='最大推力')
    ax5.set_xlabel('时间 (s)')
    ax5.set_ylabel('推力 (N)')
    ax5.set_title('控制输入')
    ax5.legend()
    ax5.grid(True)
    
    # 6. 速度跟踪
    ax6 = fig.add_subplot(2, 3, 6)
    vel_error = ref_vel_data - actual_vel_data
    vel_error_magnitude = np.linalg.norm(vel_error, axis=1)
    ax6.plot(time_data, vel_error_magnitude, 'm-', linewidth=2, label='速度误差大小')
    ax6.set_xlabel('时间 (s)')
    ax6.set_ylabel('速度误差 (m/s)')
    ax6.set_title('速度跟踪误差')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 计算性能指标
    pos_rmse = np.sqrt(np.mean(np.sum(error_data**2, axis=1)))
    max_error = np.max(np.linalg.norm(error_data, axis=1))
    final_error = np.linalg.norm(error_data[-1])
    
    print(f"\n=== 控制性能指标 ===")
    print(f"位置RMSE: {pos_rmse:.4f} m")
    print(f"最大位置误差: {max_error:.4f} m")
    print(f"最终位置误差: {final_error:.4f} m")
    print(f"平均推力: {np.mean(thrust_magnitude):.2f} N")
    print(f"最大推力: {np.max(thrust_magnitude):.2f} N")

if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\n仿真被用户中断")
    except Exception as e:
        print(f"仿真出错: {e}")
        import traceback
        traceback.print_exc()