import os
import struct
import time
import select

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

NAMES = {0: "LX", 1: "LY", 2: "LT", 3: "RX", 4: "RY", 5: "RT"}

def normalize(val):
    return max(-1.0, min(1.0, val / 32767.0))

def main(device_path: str = "/dev/input/js0"):
    fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
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
                name = NAMES.get(i, f"A{i}")
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