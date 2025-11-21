import os
import struct
import time
import select

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

NAMES_DEFAULT = {0: "LX", 1: "LY", 2: "LT", 3: "RX", 4: "RY", 5: "RT"}

def normalize(val):
    return max(-1.0, min(1.0, val / 32767.0))

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
    mode_env = os.environ.get("JOYSTICK_BT", "").lower()
    if mode_env:
        kind = "bluetooth" if mode_env in ("1", "true", "yes") else "usb"
    else:
        kind = _detect_connection_kind(device_path)

    # 构建名称映射
    names = NAMES_DEFAULT.copy()
    if kind == "bluetooth":
        # 交换 LT 与 RY 的标注
        names = {0: "LX", 1: "LY", 4: "LT", 3: "RX", 2: "RY", 5: "RT"}

    axes = {}
    dt = 0.05
    try:
        while True:
            r, _, _ = select.select([fd], [], [], 0)
            if r:
                data = os.read(fd, 8)
                if len(data) == 8:
                    time_ms, value, etype, number = struct.unpack("IhBB", data)
                    if etype & JS_EVENT_AXIS:
                        axes[number] = value
            labels = []
            for i in sorted(axes.keys()):
                name = names.get(i, f"A{i}")
                val = axes[i]
                norm = normalize(val)
                labels.append(f"{name}:{norm:+.3f}")
            line = " ".join(labels) if labels else "no axis data"
            print(line, end="\r", flush=True)
            time.sleep(dt)
    finally:
        os.close(fd)

if __name__ == "__main__":
    main()