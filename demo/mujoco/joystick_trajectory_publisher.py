import os
import struct
import time
import math
import select
from trajectory_publisher import TrajectoryPublisher

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

def normalize_axis(val):
    return max(-1.0, min(1.0, val / 32767.0))

DEADZONE = 0.05
def apply_deadzone(x):
    return 0.0 if abs(x) < DEADZONE else x

def _detect_connection_kind(device_path: str) -> str:
    try:
        base = os.path.basename(device_path)
        link = os.path.realpath(f"/sys/class/input/{base}")
        input_dir = os.path.dirname(link)
        bustype_path = os.path.join(input_dir, "id", "bustype")
        with open(bustype_path, "r") as f:
            v = f.read().strip()
        if v in ("0005", "5"):
            return "bluetooth"
        if v in ("0003", "3"):
            return "usb"
        return ""
    except Exception:
        return ""

def main(device_path: str = "/dev/input/js0"):
    fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
    pub = TrajectoryPublisher(endpoint="tcp://127.0.0.1:5555", topic="traj")
    pub.start()

    mode_env = os.environ.get("JOYSTICK_BT", "").lower()
    if mode_env:
        kind = "bluetooth" if mode_env in ("1", "true", "yes") else "usb"
    else:
        kind = _detect_connection_kind(device_path)
    axes_idx = {"LX": 0, "LY": 1, "LT": 2, "RX": 3, "RY": 4, "RT": 5}
    if kind == "bluetooth":
        axes_idx["LT"], axes_idx["RY"], axes_idx["RX"] = 4, 3, 2

    axes = {axes_idx["LX"]: 0.0, axes_idx["LY"]: 0.0, axes_idx["RX"]: 0.0, axes_idx["RY"]: 0.0, axes_idx["LT"]: 0.0, axes_idx["RT"]: 0.0}
    x, y, z = 0.0, 0.0, 0.8
    yaw = 0.0
    theta = 0.0
    move_speed = 0.7
    strafe_speed = 0.7
    yaw_speed = 0.5
    climb_speed = 0.7
    arm_speed = 0.1
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
            lx = apply_deadzone(axes.get(axes_idx["LX"], 0.0))
            ly = apply_deadzone(axes.get(axes_idx["LY"], 0.0))
            rx = apply_deadzone(axes.get(axes_idx["RX"], 0.0))
            ry = apply_deadzone(axes.get(axes_idx["RY"], 0.0))
            lt = apply_deadzone(axes.get(axes_idx["LT"], 0.0))
            rt = apply_deadzone(axes.get(axes_idx["RT"], 0.0))

            z += (-ly) * climb_speed * dt
            yaw -= lx * yaw_speed * dt
            v_f = (-ry) * move_speed
            v_r = (-rx) * strafe_speed
            x += (v_f * math.cos(yaw) - v_r * math.sin(yaw)) * dt
            y += (v_f * math.sin(yaw) + v_r * math.cos(yaw)) * dt
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