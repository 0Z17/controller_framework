import yaml
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from threading import Lock
from pathlib import Path
from datetime import datetime



class ParameterManager(ABC):
    """无人机控制器参数管理抽象类
    
    功能包括：
    1. 存储无人机控制器的各种参数
    2. 提供参数的实时引用访问
    3. 支持YAML文件的读取和保存
    """
    
    def __init__(self):
        self._lock = Lock()
        self._parameters: Dict[str, Any] = {}
        self._config_dir_path: Optional[str] = None  # 改为目录路径
        
        # 初始化默认参数
        self._initialize_default_parameters()
    
    @abstractmethod
    def _initialize_default_parameters(self):
        """初始化默认参数 - 子类必须实现"""
        pass
    
    def get_parameter(self, name: str) -> Any:
        """获取参数值（对于可变对象返回引用，支持实时更新）"""
        with self._lock:
            if name not in self._parameters:
                raise KeyError(f"参数 '{name}' 不存在")
            return self._parameters[name]
    
    def set_parameter(self, name: str, value: Any):
        """设置参数值"""
        with self._lock:
            if name not in self._parameters:
                raise KeyError(f"参数 '{name}' 不存在")
            
            # 对于numpy数组，如果形状相同则原地更新，保持引用
            if (isinstance(self._parameters[name], np.ndarray) and 
                isinstance(value, np.ndarray) and 
                self._parameters[name].shape == value.shape):
                self._parameters[name][:] = value
            else:
                # 其他情况直接替换
                self._parameters[name] = value
    
    def set_config_dir_path(self, dir_path: str):
        """设置配置文件目录路径"""
        self._config_dir_path = dir_path

    def get_all_parameters(self) -> Dict[str, Any]:
        """获取所有参数的副本"""
        with self._lock:
            result = {}
            for name, value in self._parameters.items():
                if isinstance(value, np.ndarray):
                    result[name] = value.copy()
                elif isinstance(value, (list, dict)):
                    result[name] = value.copy() if hasattr(value, 'copy') else value
                else:
                    result[name] = value
            return result
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """批量更新参数"""
        for name, value in parameters.items():
            if name in self._parameters:
                self.set_parameter(name, value)
            else:
                print(f"警告: 参数 '{name}' 不存在，跳过更新")
    
    def load_from_yaml(self, file_name: Optional[str] = None):
        """从YAML文件加载参数
        
        Args:
            file_name: 文件名（如果为None，则从默认目录加载最新的配置文件）
        """
        try:
            if file_name is None:
                # 如果未指定文件名，则查找最新的配置文件
                if self._config_dir_path is None:
                    # 使用当前文件所在目录
                    search_dir = Path(__file__).parent
                else:
                    search_dir = Path(self._config_dir_path)
                
                # 查找所有匹配的配置文件
                config_files = list(search_dir.glob("drone_config_*.yaml"))
                if not config_files:
                    raise FileNotFoundError(f"在目录 {search_dir} 中未找到配置文件")
                
                # 按文件名排序，获取最新的文件（时间戳最大的）
                config_files.sort(key=lambda x: x.name)
                file_path = config_files[-1]
                print(f"自动选择最新配置文件: {file_path.name}")
            else:
                # 指定了文件名
                if self._config_dir_path is None:
                    # 如果未指定目录，使用当前文件所在目录
                    search_dir = Path(__file__).parent
                else:
                    search_dir = Path(self._config_dir_path)
                
                file_path = search_dir / file_name
            
            if not file_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if yaml_data is None:
                print("警告: YAML文件为空")
                return
            
            # 转换numpy数组
            yaml_data = self._convert_yaml_arrays(yaml_data)
            
            # 更新参数
            self.update_parameters(yaml_data)
            
            print(f"成功从 {file_path} 加载参数")
            return str(file_path)  # 返回实际加载的文件路径
            
        except Exception as e:
            print(f"加载YAML文件失败: {e}")
            raise
    
    def save_to_yaml(self, file_name: Optional[str] = None, readable_format: bool = True):
        """保存参数到YAML文件
        
        Args:
            file_name: 保存文件名（如果为None，则使用默认目录+时间戳命名）
            readable_format: 是否使用可读性更好的格式
        
        Returns:
            str: 实际保存的文件路径
        """
        try:
            # 如果默认路径为空，则保存到当前文件所在目录
            if file_name is None:
                if self._config_dir_path is None:
                    # 使用当前文件所在目录
                    save_dir = Path(__file__).parent
                else:
                    save_dir = Path(self._config_dir_path)
                
                # 生成时间戳文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"drone_config_{timestamp}.yaml"
                file_path = save_dir / filename
            else:
                if self._config_dir_path is None:
                    # 如果未指定目录，也使用当前文件所在目录
                    save_dir = Path(__file__).parent
                else:
                    save_dir = Path(self._config_dir_path)
                file_path = save_dir / file_name
            
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if readable_format:
                self._save_readable_yaml(file_path)
            else:
                # 使用标准格式
                save_data = self._prepare_yaml_data()
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(save_data, f, default_flow_style=False, 
                             allow_unicode=True, sort_keys=True)
            
            print(f"成功保存参数到 {file_path}")
            return str(file_path)  # 返回实际保存的文件路径
            
        except Exception as e:
            print(f"保存YAML文件失败: {e}")
            raise
    
    def _save_readable_yaml(self, file_path: Path):
        """保存为可读性更好的YAML格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# 无人机控制器参数配置文件\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# 自动生成，请谨慎修改\n\n")
            
            # 按类别组织参数
            categories = {
                '控制器参数': ['position_kp', 'position_ki', 'position_kd', 
                              'attitude_kp', 'attitude_ki', 'attitude_kd',
                              'angular_rate_kp', 'angular_rate_ki', 'angular_rate_kd'],
                '物理参数': ['mass', 'gravity', 'inertia_matrix'],
                '限制参数': ['max_velocity', 'max_acceleration', 'max_angular_velocity', 
                           'max_thrust', 'max_torque'],
                '控制器配置': ['control_frequency', 'enable_integral_windup_protection', 
                            'integral_limit', 'deadzone_threshold'],
            }
            
            for category, param_names in categories.items():
                f.write(f"# {category}\n")
                for param_name in param_names:
                    if param_name in self._parameters:
                        value = self._parameters[param_name]
                        self._write_parameter(f, param_name, value)
                f.write("\n")
    
    def _write_parameter(self, f, name: str, value: Any):
        """写入单个参数到文件"""
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                # 一维数组，写成一行
                f.write(f"{name}: [{', '.join(map(str, value))}]\n")
            elif value.ndim == 2:
                # 二维矩阵，写成矩阵格式
                f.write(f"{name}:\n")
                for row in value:
                    f.write(f"  - [{', '.join(f'{x:8.3f}' for x in row)}]\n")
            else:
                # 高维数组，使用标准格式
                f.write(f"{name}: {value.tolist()}\n")
        elif isinstance(value, bool):
            f.write(f"{name}: {str(value).lower()}\n")
        else:
            f.write(f"{name}: {value}\n")
    
    def _convert_yaml_arrays(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """将YAML中的列表转换为numpy数组（如果需要）"""
        result = {}
        for key, value in data.items():
            if key in self._parameters and isinstance(self._parameters[key], np.ndarray):
                if isinstance(value, list):
                    result[key] = np.array(value, dtype=self._parameters[key].dtype)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def _prepare_yaml_data(self) -> Dict[str, Any]:
        """准备用于保存到YAML的数据"""
        save_data = {}
        for key, value in self._parameters.items():
            if isinstance(value, np.ndarray):
                # 将numpy数组转换为列表
                save_data[key] = value.tolist()
            else:
                save_data[key] = value
        return save_data
    
    def list_parameters(self) -> List[str]:
        """列出所有参数名称"""
        return list(self._parameters.keys())
    
    def parameter_exists(self, name: str) -> bool:
        """检查参数是否存在"""
        return name in self._parameters
    
    def reset_to_defaults(self):
        """重置所有参数到默认值"""
        with self._lock:
            self._parameters.clear()
            self._initialize_default_parameters()
    
    def get_parameter_info(self, name: str) -> Dict[str, Any]:
        """获取参数信息"""
        if not self.parameter_exists(name):
            raise KeyError(f"参数 '{name}' 不存在")
        
        value = self._parameters[name]
        return {
            'name': name,
            'type': type(value).__name__,
            'shape': getattr(value, 'shape', None),
            'dtype': getattr(value, 'dtype', None),
            'value': value
        }
