"""
Control Framework - UAV控制框架

这个包提供了无人机控制系统的基础组件，包括：
- 控制器基类和几何控制器
- 参数管理器
- 状态接口管理
- 数学工具函数
"""

# 导入主要的类和函数
from .controller_base import ControllerBase, ControllerOutput
from .parameter_manager import ParameterManager
from .state_interface import StateManager
from .math_utils import MathUtils

# 定义包的公开接口
__all__ = [
    # 控制器相关
    'ControllerBase',
    'ControllerOutput', 
    
    # 管理器相关
    'ParameterManager',
    'StateManager',
    
    # 数学工具（从 math_utils 导入的所有函数）
    'MathUtils',
]

# 版本信息
__version__ = "0.1.0"
__author__ = "Control Framework Team"