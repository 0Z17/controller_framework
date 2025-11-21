import os
import sys
import numpy as np
import mujoco as mj
import zmq
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from control_framework.math_utils import MathUtils
import json

class DummyRobot:
    def __init__(self, model_path: str = "scene.xml"):
        if os.path.isabs(model_path):
            path = model_path
        else:
            path = os.path.join(os.path.dirname(__file__), model_path)
        self.model = mj.MjModel.from_xml_path(path)
        self.data = mj.MjData(self.model)
        self.body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base_link_dummy")
        if self.body_id < 0:
            raise ValueError("base_link_dummy body not found")
        self.mocap_id = int(self.model.body_mocapid[self.body_id])
        if self.mocap_id < 0:
            raise ValueError("base_link_dummy is not a mocap body")
        self.operator_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "operator_Link_dummy")
        if self.operator_body_id < 0:
            raise ValueError("operator_Link_dummy body not found")
        self.operator_mocap_id = int(self.model.body_mocapid[self.operator_body_id])
        if self.operator_mocap_id < 0:
            raise ValueError("operator_Link_dummy is not a mocap body")
        self.operator_rel_pos = np.array([0.000870575, 0.000171621, 0.0284551], dtype=float)
        self._zmq_ctx = None
        self._zmq_sock = None

    def set_base_pose(self, position: np.ndarray, quaternion: np.ndarray):
        q = MathUtils.normalize_quaternion(np.asarray(quaternion).astype(float))
        p = np.asarray(position).astype(float)
        self.data.mocap_pos[self.mocap_id] = p
        self.data.mocap_quat[self.mocap_id] = q
        mj.mj_forward(self.model, self.data)

    def set_base_pose_euler(self, position: np.ndarray, euler_rpy: np.ndarray):
        e = np.asarray(euler_rpy).astype(float)
        R = MathUtils.euler_to_rotation_matrix(e[0], e[1], e[2])
        q = MathUtils.rotation_matrix_to_quaternion(R)
        self.set_base_pose(position, q)

    def set_base_position(self, position: np.ndarray):
        _, quat = self.get_base_pose()
        p = np.asarray(position).astype(float)
        self.set_base_pose(p, quat)

    def set_base_quaternion(self, quaternion: np.ndarray):
        pos, _ = self.get_base_pose()
        q = MathUtils.normalize_quaternion(np.asarray(quaternion).astype(float))
        self.set_base_pose(pos, q)

    def get_base_pose(self):
        pos = self.data.mocap_pos[self.mocap_id].copy()
        quat = self.data.mocap_quat[self.mocap_id].copy()
        return pos, quat

    def set_operator_pose_relative(self, rel_pos_body: np.ndarray, rel_quat_body: np.ndarray):
        base_pos, base_quat = self.get_base_pose()
        q = MathUtils.normalize_quaternion(base_quat)
        v = np.array([0.0, *np.asarray(rel_pos_body).astype(float)])
        v_rot = MathUtils.quaternion_multiply(MathUtils.quaternion_multiply(q, v), MathUtils.quaternion_conjugate(q))
        p_world = base_pos + v_rot[1:]
        q_world = MathUtils.quaternion_multiply(base_quat, np.asarray(rel_quat_body).astype(float))
        q_world = MathUtils.normalize_quaternion(q_world)
        self.data.mocap_pos[self.operator_mocap_id] = p_world
        self.data.mocap_quat[self.operator_mocap_id] = q_world
        mj.mj_forward(self.model, self.data)

    def set_operator_angle(self, angle: float):
        R_rel = MathUtils.euler_to_rotation_matrix(0.0, angle, 0.0)
        q_rel = MathUtils.rotation_matrix_to_quaternion(R_rel)
        self.set_operator_pose_relative(self.operator_rel_pos, q_rel)

    def set_operator_offset(self, rel_pos_body: np.ndarray):
        self.operator_rel_pos = np.asarray(rel_pos_body).astype(float)
        
    def start_trajectory_subscriber(self, endpoint: str = "tcp://127.0.0.1:5555", topic: str = "traj"):
        self._zmq_ctx = zmq.Context.instance()
        self._zmq_sock = self._zmq_ctx.socket(zmq.SUB)
        self._zmq_sock.connect(endpoint)
        self._zmq_sock.setsockopt_string(zmq.SUBSCRIBE, topic)

    def poll_and_apply_reference(self):
        if self._zmq_sock is None:
            return False
        try:
            parts = self._zmq_sock.recv_multipart(flags=zmq.NOBLOCK)
        except Exception:
            return False
        payload = parts[1].decode()
        msg = json.loads(payload)
        schema = msg.get("schema", [])
        values = msg.get("values", [])
        m = {k: float(values[i]) for i, k in enumerate(schema)}
        x = m.get("x", 0.0)
        y = m.get("y", 0.0)
        z = m.get("z", 0.0)
        yaw = m.get("yaw", 0.0)
        theta = m.get("theta", 0.0)
        R = MathUtils.euler_to_rotation_matrix(0.0, 0.0, yaw)
        q = MathUtils.rotation_matrix_to_quaternion(R)
        self.set_base_pose(np.array([x, y, z], dtype=float), q)
        self.set_operator_angle(theta)
        return True

    def close_trajectory_subscriber(self):
        if self._zmq_sock is not None:
            self._zmq_sock.close(0)
            self._zmq_sock = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
