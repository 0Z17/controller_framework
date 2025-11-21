import zmq
import json
from typing import List, Union, Dict

class TrajectoryPublisher:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555", topic: str = "traj", schema: List[str] = None):
        self.endpoint = endpoint
        self.topic = topic
        self.schema = schema or [
            "x","y","z","yaw","theta",
            "vx","vy","vz","vyaw","vtheta",
            "ax","ay","az","ayaw","atheta"
        ]
        self.ctx = None
        self.sock = None

    def start(self):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(self.endpoint)

    def publish(self, signal: Union[List[float], Dict[str, float]]):
        if isinstance(signal, dict):
            values = [float(signal.get(k, 0.0)) for k in self.schema]
        else:
            values = [float(v) for v in signal]
        msg = json.dumps({"schema": self.schema, "values": values})
        self.sock.send_multipart([self.topic.encode(), msg.encode()])

    def close(self):
        if self.sock is not None:
            self.sock.close(0)
            self.sock = None
        if self.ctx is not None:
            self.ctx.term()
            self.ctx = None