import os
import sys
import time
import math
import numpy as np
import mujoco as mj
import mujoco.viewer as viewer

sys.path.append(os.path.dirname(__file__))
from dummy_robot import DummyRobot
from control_framework.math_utils import MathUtils

def main():
    dr = DummyRobot('scene.xml')
    v = viewer.launch_passive(dr.model, dr.data)
    dt = dr.model.opt.timestep
    dr.start_trajectory_subscriber(endpoint="tcp://127.0.0.1:5555", topic="traj")

    while v.is_running():
        dr.poll_and_apply_reference()
        v.sync()
        time.sleep(0.01)

    dr.close_trajectory_subscriber()

if __name__ == '__main__':
    main()