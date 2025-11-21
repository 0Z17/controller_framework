import sys
import termios
import tty
import select
import time
import math
from trajectory_publisher import TrajectoryPublisher

def main():
    pub = TrajectoryPublisher(endpoint="tcp://127.0.0.1:5555", topic="traj")
    pub.start()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    x, y, z = 0.0, 0.0, 0.8
    yaw = 0.0
    theta = 0.0
    lin_speed = 1.0
    ang_speed = 1.5
    arm_speed = 1.2
    climb_speed = 0.8
    dt = 0.02
    try:
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == 'i':
                    x += lin_speed*dt*math.cos(yaw)
                    y += lin_speed*dt*math.sin(yaw)
                elif ch == 'k':
                    x -= lin_speed*dt*math.cos(yaw)
                    y -= lin_speed*dt*math.sin(yaw)
                elif ch == 'j':
                    x += -lin_speed*dt*math.sin(yaw)
                    y += lin_speed*dt*math.cos(yaw)
                elif ch == 'l':
                    x += lin_speed*dt*math.sin(yaw)
                    y += -lin_speed*dt*math.cos(yaw)
                elif ch == 'u':
                    yaw -= ang_speed*dt
                elif ch == 'o':
                    yaw += ang_speed*dt
                elif ch == 'r':
                    z += climb_speed*dt
                elif ch == 'f':
                    z -= climb_speed*dt
                elif ch == 'n':
                    theta -= arm_speed*dt
                elif ch == 'm':
                    theta += arm_speed*dt
                elif ch == '+':
                    lin_speed *= 1.1
                elif ch == '-':
                    lin_speed /= 1.1
                elif ch == '\x03':
                    break
            pub.publish({
                "x": x, "y": y, "z": z, "yaw": yaw, "theta": theta,
                "vx": 0.0, "vy": 0.0, "vz": 0.0, "vyaw": 0.0, "vtheta": 0.0,
                "ax": 0.0, "ay": 0.0, "az": 0.0, "ayaw": 0.0, "atheta": 0.0,
            })
            time.sleep(dt)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        pub.close()

if __name__ == '__main__':
    main()