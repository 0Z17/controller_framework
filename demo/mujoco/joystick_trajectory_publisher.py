import os
import struct
import time
import math
import select
from trajectory_publisher import TrajectoryPublisher

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

AXIS_LX = 0
AXIS_LY = 1
AXIS_LT = 2
AXIS_RX = 3
AXIS_RY = 4
AXIS_RT = 5

def normalize_axis(val):
    return max(-1.0, min(1.0, val / 32767.0))

DEADZONE = 0.05
def apply_deadzone(x):
    return 0.0 if abs(x) < DEADZONE else x

def main(device_path: str = "/dev/input/js0"):
    fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
    pub = TrajectoryPublisher(endpoint="tcp://127.0.0.1:5555", topic="traj")
    pub.start()

    axes = {AXIS_LX: 0.0, AXIS_LY: 0.0, AXIS_RX: 0.0, AXIS_RY: 0.0, AXIS_LT: 0.0, AXIS_RT: 0.0}
    x, y, z = 0.0, 0.0, 0.8
    yaw = 0.0
    theta = 0.0
    move_speed = 1.5
    strafe_speed = 1.5
    yaw_speed = 1.5
    climb_speed = 1.0
    arm_speed = 1.2
    dt = 0.02

    try:
        while True:
            r, _, _ = select.select([fd], [], [], 0)
            if r:
                data = os.read(fd, 8)
                if len(data) == 8:
                    time_ms, value, etype, number = struct.unpack('IhBB', data)
                    if etype & JS_EVENT_AXIS:
                        axes[number] = normalize_axis(value)
            lx = apply_deadzone(axes.get(AXIS_LX, 0.0))
            ly = apply_deadzone(axes.get(AXIS_LY, 0.0))
            rx = apply_deadzone(axes.get(AXIS_RX, 0.0))
            ry = apply_deadzone(axes.get(AXIS_RY, 0.0))
            lt = apply_deadzone(axes.get(AXIS_LT, 0.0))
            rt = apply_deadzone(axes.get(AXIS_RT, 0.0))

            z += (-ly) * climb_speed * dt
            yaw -= lx * yaw_speed * dt
            x += (-ry) * move_speed * dt * math.cos(yaw) + (rx) * strafe_speed * dt * (-math.sin(yaw))
            y -= (-ry) * move_speed * dt * math.sin(yaw) + (rx) * strafe_speed * dt * (math.cos(yaw))
            theta += (rt - lt) * arm_speed * dt

            pub.publish({
                "x": x, "y": y, "z": z, "yaw": yaw, "theta": theta,
                "vx": 0.0, "vy": 0.0, "vz": 0.0, "vyaw": 0.0, "vtheta": 0.0,
                "ax": 0.0, "ay": 0.0, "az": 0.0, "ayaw": 0.0, "atheta": 0.0,
            })
            time.sleep(dt)
    finally:
        pub.close()
        os.close(fd)

if __name__ == '__main__':
    main()