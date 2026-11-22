"""Stand-in for the Go2's sport service in the MuJoCo sim.

Listens on rt/api/sport/request (what the starter's action_bridge publishes as /api/sport/request), acks on
rt/api/sport/response and plays the matching canned trajectory from sport_actions.py through rt/lowcmd.

Start the simulator first (see sim/mujoco/README.md), then: python sim/mujoco/sport_emulator.py
"""
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, RequestIdentity_, Response_, ResponseHeader_, ResponseStatus_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from sport_actions import ACTIONS, KD, KP, STARTUP, Trajectory

SIM_DOMAIN_ID = 1  # real Go2 uses 0; keep the sim separate
UNKNOWN_API_CODE = 3203  # any non-zero code makes action_bridge free itself as "rejected"
CONTROL_DT = 0.002


class SportEmulator:
    def __init__(self):
        self.lock = threading.Lock()
        self.joints = None
        self.trajectory = None
        self.started = 0.0

        self.response_pub = ChannelPublisher("rt/api/sport/response", Response_)
        self.response_pub.Init()
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self.on_state, 10)
        self.request_sub = ChannelSubscriber("rt/api/sport/request", Request_)
        self.request_sub.Init(self.on_request, 10)

    def on_state(self, msg):
        self.joints = np.array([msg.motor_state[i].q for i in range(12)])

    def play(self, frames):
        with self.lock:
            now = time.perf_counter()
            start = self.trajectory.target(now - self.started) if self.trajectory else self.joints
            self.trajectory = Trajectory(start, frames)
            self.started = now

    def on_request(self, msg):
        api_id = msg.header.identity.api_id
        code = 0
        if api_id in ACTIONS:
            name, frames = ACTIONS[api_id]
            self.play(frames)
            print(f"api_id={api_id} ({name}), {self.trajectory.duration:.1f}s", flush=True)
        else:
            code = UNKNOWN_API_CODE
            print(f"api_id={api_id} not emulated, rejected", flush=True)
        # built by hand: unitree_sdk2py's default Response_ helper raises a TypeError
        identity = RequestIdentity_(msg.header.identity.id, api_id)
        response = Response_(ResponseHeader_(identity, ResponseStatus_(code)), "", [])
        self.response_pub.Write(response)

    def run(self):
        while self.joints is None:
            print("waiting for rt/lowstate - is the simulator running?", flush=True)
            time.sleep(1.0)
        self.play(STARTUP)
        print("sport emulator ready, standing up", flush=True)

        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0], cmd.head[1], cmd.level_flag = 0xFE, 0xEF, 0xFF
        crc = CRC()
        while True:
            with self.lock:
                target = self.trajectory.target(time.perf_counter() - self.started)
            for i in range(12):
                motor = cmd.motor_cmd[i]
                motor.mode, motor.q, motor.kp, motor.kd = 0x01, float(target[i]), KP, KD
            cmd.crc = crc.Crc(cmd)
            self.cmd_pub.Write(cmd)
            time.sleep(CONTROL_DT)


if __name__ == "__main__":
    ChannelFactoryInitialize(SIM_DOMAIN_ID, "lo")
    try:
        SportEmulator().run()
    except KeyboardInterrupt:
        pass
