"""Minimal low-level controller used to test supervisor.py: stand up from wherever the legs are and
hold the stand pose (500 Hz LowCmd with CRC). Optional fault injection for the tests.

  python supervisor.py --domain 77 --iface lo -- python stand_hold.py --domain 77 --iface lo
  python stand_hold.py ... --fault crash --fault-at 3     # crash | hang | nan | wild | badcrc | slow

This is a test fixture, not the dodge controller. On the real robot pass --release-sport-mode: it lies
the dog down and releases sport mode exactly like unitree_sdk2_python's go2_stand_example.py (not
testable in the fake robot, which has no motion-switcher service).
"""
from __future__ import annotations

import argparse
import math
import os
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# SDK order FR, FL, RR, RL (= walk-these-ways' default pose, reordered)
STAND_Q = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]
KP, KD, RAMP = 50.0, 3.0, 1.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--fault", default="none", choices=["none", "crash", "hang", "nan", "wild", "badcrc", "slow"])
    ap.add_argument("--fault-at", type=float, default=3.0, help="seconds after the first command")
    ap.add_argument("--release-sport-mode", action="store_true", help="real robot: StandDown + ReleaseMode first")
    a = ap.parse_args()
    ChannelFactoryInitialize(a.domain, a.iface)
    if a.release_sport_mode:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        sc, msc = SportClient(), MotionSwitcherClient()
        for c in (sc, msc):
            c.SetTimeout(5.0)
            c.Init()
        status, result = msc.CheckMode()
        while result["name"]:
            sc.StandDown()
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1)
        print("[stand_hold] sport mode released", flush=True)
    state = {}
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda m: state.__setitem__("q", [x.q for x in m.motor_state[:12]]), 0)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0], cmd.head[1], cmd.level_flag = 0xFE, 0xEF, 0xFF
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
    while "q" not in state:
        time.sleep(0.01)
    q0 = list(state["q"])
    print(f"[stand_hold] got state, standing up (fault {a.fault} at {a.fault_at}s)", flush=True)
    t_start = time.monotonic()
    next_t = t_start
    while True:
        t = time.monotonic() - t_start
        s = min(1.0, t / RAMP)
        s = 0.5 - 0.5 * math.cos(math.pi * s)
        faulty = a.fault != "none" and t >= a.fault_at
        if faulty and a.fault == "crash":
            print("[stand_hold] injected crash", flush=True)
            os._exit(3)
        if faulty and a.fault == "hang":
            print("[stand_hold] injected hang", flush=True)
            while True:
                time.sleep(1)
        for i in range(12):
            m = cmd.motor_cmd[i]
            m.q = (1 - s) * q0[i] + s * STAND_Q[i]
            m.dq, m.tau, m.kp, m.kd = 0.0, 0.0, KP, KD
        if faulty and a.fault == "nan":
            cmd.motor_cmd[4].q = float("nan")
        if faulty and a.fault == "wild":
            cmd.motor_cmd[2].q = 0.5            # FR calf beyond its range (-2.72..-0.84)
        cmd.crc = crc.Crc(cmd)
        if faulty and a.fault == "badcrc":
            cmd.crc ^= 0x1
        pub.Write(cmd)
        next_t += 0.002 if not (faulty and a.fault == "slow") else 0.08    # 'slow': 80 ms gaps
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.monotonic()


if __name__ == "__main__":
    main()
