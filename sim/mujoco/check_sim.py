"""Smoke test for the unitree_mujoco Go2 sim: stand up via rt/lowcmd, report body height from rt/sportmodestate.

Start the simulator first (see sim/mujoco/README.md), then: python sim/mujoco/check_sim.py
"""
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, SportModeState_
from unitree_sdk2py.utils.crc import CRC

SIM_DOMAIN_ID = 1  # real Go2 uses 0; keep the sim separate
STAND = np.array([0.0057, 0.6088, -1.2176, -0.0057, 0.6088, -1.2176] * 2)
DOWN = np.array([0.0473, 1.2219, -2.4438, -0.0473, 1.2219, -2.4438] * 2)

heights = []
ChannelFactoryInitialize(SIM_DOMAIN_ID, "lo")
sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
sub.Init(lambda msg: heights.append(msg.position[2]), 10)
pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()

cmd = unitree_go_msg_dds__LowCmd_()
cmd.head[0], cmd.head[1], cmd.level_flag = 0xFE, 0xEF, 0xFF
for m in cmd.motor_cmd:
    m.mode = 0x01
crc = CRC()

time.sleep(1.0)
if not heights:
    raise SystemExit("FAIL: no rt/sportmodestate received - is the simulator running on domain 1 / lo?")
start = heights[-1]

t0 = time.perf_counter()
while (t := time.perf_counter() - t0) < 4.0:
    phase = np.tanh(t / 1.2)
    for i in range(12):
        cmd.motor_cmd[i].q = phase * STAND[i] + (1 - phase) * DOWN[i]
        cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
        cmd.motor_cmd[i].kd = 3.5
    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)
    time.sleep(0.002)

end = heights[-1]
print(f"state msgs: {len(heights)}, body height {start:.3f} m -> {end:.3f} m")
print("OK" if end - start > 0.1 else "FAIL: robot did not stand up")
