import numpy as np
import sys
import os

from control_framework import MathUtils, ControllerBase, ControllerOutput

# 添加src目录到路径
sys.path.append(os.path.dirname(__file__))

# from controller_base import ControllerBase, ControllerOutput
# from parameter_manager import ParameterManager
# from state_interface import StateManager

class GeometryController(ControllerBase):
    """几何控制器"""

    def _initialize_controller(self):
        self.K_pos = self._parameter_manager.get_parameter('K_pos').copy()
        self.K_vel = self._parameter_manager.get_parameter('K_vel').copy()
        self.K_ori = self._parameter_manager.get_parameter('K_ori').copy()
        self.K_ang_vel = self._parameter_manager.get_parameter('K_ang_vel').copy()
        self.m = self._parameter_manager.get_parameter('mass')
        self.J = self._parameter_manager.get_parameter('inertia_matrix').copy()
        self.g = self._parameter_manager.get_parameter('gravity')
    

    def _compute_control(self):
        # get states
        ref_acc_body = self.get_ref_acc_body()
        ref_ang_acc_body = self.get_ref_ang_acc_body()
        R = MathUtils.quaternion_to_rotation_matrix(self.get_ori_body())
        vel = self.get_vel_body()
        ang_vel = self.get_ang_vel_body()
        e3 = np.array([0, 0, 1])
        time_stamp = self.get_timestamp()

        # get state errors
        e_p = self._state_manager.get_pos_err()
        e_v = self._state_manager.get_vel_err()
        e_ori = self._state_manager.get_ori_body_err_so3()
        e_ang_vel = self._state_manager.get_ang_vel_body_err_so3()

        # get fictitious input
        v_pos = ref_acc_body + R.T @ (self.K_pos * e_p + self.K_vel * e_v)
        v_ori = ref_ang_acc_body + self.K_ori * e_ori + self.K_ang_vel * e_ang_vel

        # get the force and torque outputs
        force = self.m * v_pos + self.m * self.g * (R.T @ e3) - np.cross(ang_vel, vel)
        torque = self.J @ v_ori - np.cross(ang_vel, self.J @ ang_vel)

        self._output.force = force
        self._output.torque = torque
        self._output.timestamp = time_stamp
