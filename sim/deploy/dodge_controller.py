"""Our low-level controller for the real Go2: stand up, hold, run the SAFE dodge on command, lie down.

Never start this by hand: go2_modes.py starts it as the child of supervisor.py once sport mode is
released. Commands arrive on stdin, one per line (go2_modes.py writes them):
  dodge left | dodge right    (plain "dodge" = left)
          run the SAFE dodge, exactly the sequence validated in sim (results/safe_dodge.json; the right
          one is its mirror, all sideways commands negated, results/safe_dodge_right.json):
          0.5 s stand, ~2 s trotting in place, kick + dodge sideways, brake, 1.5 s back to stand.
          Only accepted from a settled stand.
  down    lie down (after a running dodge has finished) and exit with code 0.
  EOF     (go2_modes.py died) = down.

  --gentle  all sideways speed commands x0.6 (kick 0.82 m/s): 300 harsh sim runs, 0 falls, peak tilt
          11 deg (SAFE: 17), ends 24-30 cm left. For the first tries on the real dog.

Joint targets are clamped 0.1 rad inside the joint range, as in the harsh-physics validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import go2_dodge_sim as g  # noqa: E402
import optimize_dodge as o  # noqa: E402
import safety  # noqa: E402

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

PERIOD = 0.002
POLICY_TO_SDK = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])    # FL FR RL RR <-> FR FL RR RL (its own inverse)
STAND = g.DEFAULT_Q[POLICY_TO_SDK]                                  # SDK order
# Unitree go2_stand_example.py: folded pose (stand-up via this) and lie-down pose, gains 60 / 5
FOLDED = np.array([0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65])
LIE = np.array([-0.35, 1.36, -2.65, 0.35, 1.36, -2.65, -0.5, 1.36, -2.65, 0.5, 1.36, -2.65])
MOVE_KP, MOVE_KD = 60.0, 5.0
FOLD_S, RISE_S, GAIN_S, LIE_S, LIE_HOLD_S = 1.0, 1.5, 0.5, 2.0, 0.5
MARGIN = 0.1
Q_LO, Q_HI = safety.Q_RANGE[:, 0] + MARGIN, safety.Q_RANGE[:, 1] - MARGIN
SETTLED_S, SETTLED_TILT, SETTLED_ERR = 1.0, 0.15, 0.2     # dodge only from this calm a stand


GENTLE_SCALE = 0.6


def load_safe_dodge(gentle: bool = False, side: str = "left") -> g.Dodge:
    """side='right' mirrors the dodge: all sideways commands negated (+vy = left)."""
    p = dict(json.loads((HERE.parent / "results/safe_dodge.json").read_text())["chosen"]["params"])
    s = (GENTLE_SCALE if gentle else 1.0) * (-1.0 if side == "right" else 1.0)
    for k in ("kick_vy", "hold_vy", "brake_vy"):   # same as validate_dodge.scaled(p, s)
        p[k] *= s
    return o.to_dodge(p, ("SAFE x0.6 (gentle)" if gentle else "SAFE") + f" {side}")


def smooth(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return 0.5 - 0.5 * math.cos(math.pi * s)


class RealRobot:
    """The bits of go2_dodge_sim.Go2Robot that DodgeController uses, fed from rt/lowstate."""

    def __init__(self):
        self.lock = threading.Lock()
        self.q_sdk = self.dq_sdk = self.quat = None
        self.n = 0

    def on_state(self, msg: LowState_) -> None:
        q = np.array([m.q for m in msg.motor_state[:12]])
        dq = np.array([m.dq for m in msg.motor_state[:12]])
        quat = np.array(msg.imu_state.quaternion, dtype=float)
        with self.lock:
            self.q_sdk, self.dq_sdk, self.quat = q, dq, quat
            self.n += 1

    def snapshot(self):
        with self.lock:
            return self.q_sdk, self.dq_sdk, self.quat

    @property
    def q(self) -> np.ndarray:
        return self.snapshot()[0][POLICY_TO_SDK]

    def observe(self):
        q, dq, quat = self.snapshot()
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, quat)
        grav = r.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        return grav, q[POLICY_TO_SDK], dq[POLICY_TO_SDK]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--gentle", action="store_true", help="sideways speeds x0.6 (validated, see above)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    log = lambda s: print(f"[controller] {s}", flush=True)       # noqa: E731

    dodges = {side: load_safe_dodge(a.gentle, side) for side in ("left", "right")}
    dodge = dodges["left"]          # stand gains and timing are the same for both sides
    policy = g.WTWPolicy()
    # TorchScript optimises on the first calls (100s of ms): do that now, not mid-dodge, where a
    # stall > 50 ms trips the supervisor
    dt = []
    for _ in range(40):
        t = time.perf_counter()
        policy.targets(np.array([0.0, 0.0, -1.0]), g.DEFAULT_Q, np.zeros(12), dodge.cmd)
        dt.append(time.perf_counter() - t)
    policy.reset()
    log(f"policy warmed up: first call {dt[0] * 1000:.0f} ms, now {np.median(dt[-10:]) * 1000:.1f} ms "
        f"(max {max(dt[-10:]) * 1000:.1f} ms)")
    robot = RealRobot()
    ChannelFactoryInitialize(a.domain, a.iface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(robot.on_state, 10)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0], cmd.head[1], cmd.level_flag = 0xFE, 0xEF, 0xFF
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01

    commands: queue.Queue[str] = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            commands.put(line.strip())
        commands.put("down")                                   # EOF: go2_modes.py is gone
    threading.Thread(target=read_stdin, daemon=True).start()

    t_wait = time.monotonic()
    while robot.snapshot()[0] is None:
        if time.monotonic() - t_wait > 5:
            log("no rt/lowstate within 5 s, giving up (nothing sent)")
            sys.exit(4)
        time.sleep(0.01)
    q0 = robot.snapshot()[0].copy()
    log(f"standing up (dodge: {dodge.name} / right, kick {dodge.kick_vy:.2f} m/s for {dodge.kick_s:.2f} s, "
        f"hold {dodge.cmd.vy:.2f}, trigger {dodge.t0:.1f} s after the command)")

    mode, t_mode = "up", time.monotonic()
    want_down = False
    from_q = q0                     # ramp start for 'up' and 'down'
    ctl = hold_since = None
    target, kp, kd = q0.copy(), MOVE_KP, MOVE_KD
    next_policy = 0.0
    next_t = time.monotonic()
    while True:
        now = time.monotonic()
        tm = now - t_mode
        while not commands.empty():
            c = commands.get()
            if c == "down":
                if not want_down:
                    log("lie down requested" + (" (after the dodge)" if mode == "dodge" else ""))
                want_down = True
            elif c in ("dodge", "dodge left", "dodge right"):
                side = "right" if c == "dodge right" else "left"
                q, _, quat = robot.snapshot()
                tilt = max(map(abs, safety.tilt_from_quat(quat)))
                err = float(np.abs(q - STAND).max())
                if mode != "hold" or want_down:
                    log(f"dodge refused: not standing still ({mode})")
                elif now - hold_since < SETTLED_S or tilt > SETTLED_TILT or err > SETTLED_ERR:
                    log(f"dodge refused: not settled (stand {now - hold_since:.1f} s, tilt {tilt:.2f} rad, "
                        f"joint error {err:.2f} rad)")
                else:
                    policy.reset()
                    active = dodges[side]
                    ctl = g.DodgeController(active, robot, policy)
                    mode, t_mode, tm, next_policy = "dodge", now, 0.0, 0.0
                    worst_gap, t_prev = 0.0, now
                    log(f"DODGE: trotting in place, kick to the {side.upper()} in {active.t0:.1f} s")
            elif c:
                log(f"unknown command {c!r}")

        if want_down and mode in ("up", "hold"):
            mode, t_mode, tm, from_q = "down", now, 0.0, robot.snapshot()[0].copy()
            log("lying down")

        if mode == "up":
            if tm < FOLD_S:
                target = from_q + smooth(tm / FOLD_S) * (FOLDED - from_q)
                kp, kd = MOVE_KP, MOVE_KD
            elif tm < FOLD_S + RISE_S:
                target = FOLDED + smooth((tm - FOLD_S) / RISE_S) * (STAND - FOLDED)
                kp, kd = MOVE_KP, MOVE_KD
            elif tm < FOLD_S + RISE_S + GAIN_S:          # blend to the stand gains the dodge starts from
                s = (tm - FOLD_S - RISE_S) / GAIN_S
                target = STAND
                kp, kd = MOVE_KP + s * (dodge.stand_kp - MOVE_KP), MOVE_KD + s * (dodge.stand_kd - MOVE_KD)
            else:
                mode, t_mode, hold_since = "hold", now, now
                log("standing. Ready to dodge.")
        if mode == "hold":
            target, kp, kd = STAND, dodge.stand_kp, dodge.stand_kd
        elif mode == "dodge":
            worst_gap, t_prev = max(worst_gap, now - t_prev), now
            if tm >= next_policy:                           # 50 Hz policy step on the wall clock
                if next_policy >= active.total - 1e-9:
                    mode, t_mode, hold_since = "hold", now, now
                    log(f"dodge done, standing (longest gap between commands {worst_gap * 1000:.1f} ms)")
                    target, kp, kd = STAND, dodge.stand_kp, dodge.stand_kd
                else:
                    q_pol, kp, kd, phase = ctl.targets(next_policy)
                    target = np.asarray(q_pol)[POLICY_TO_SDK]
                    next_policy += g.POLICY_DT
                    if tm - next_policy > 0.1:              # fell far behind (should not happen): stay in step
                        log(f"policy {tm - next_policy:.2f} s behind the clock")
                        next_policy = tm
        elif mode == "down":
            target = from_q + smooth(tm / LIE_S) * (LIE - from_q)
            kp, kd = MOVE_KP, MOVE_KD
            if tm >= LIE_S + LIE_HOLD_S:
                q = robot.snapshot()[0]
                log(f"lying ({'joints folded' if safety.is_lying(q) else 'WARNING: joints not folded'}), exiting")
                sys.stdout.flush()
                # immediately: a normal interpreter shutdown (DDS, torch) takes > 50 ms, and the
                # supervisor would see 'controller silent' before it sees the clean exit
                os._exit(0)

        target = np.clip(target, Q_LO, Q_HI)
        for i in range(12):
            m = cmd.motor_cmd[i]
            m.q, m.dq, m.tau, m.kp, m.kd = float(target[i]), 0.0, 0.0, float(kp), float(kd)
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        next_t += PERIOD
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.monotonic()


if __name__ == "__main__":
    main()
