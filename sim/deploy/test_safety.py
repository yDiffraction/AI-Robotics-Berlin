"""Unit tests of safety.py (no DDS; runs on Windows too).

  python test_safety.py
Includes the false-alarm check: every command the SAFE dodge sends under harsh physics must pass
check_command, and its tilt must stay below the trip threshold.
"""
from __future__ import annotations

import math
import struct
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import go2_dodge_sim as g  # noqa: E402
import safety as S  # noqa: E402

FAILS: list[str] = []
POLICY_TO_SDK = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]      # FL FR RL RR -> FR FL RR RL


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def test_remote() -> None:
    print("remote bytes")
    # layout as in unitree_sdk2_python example/wireless_controller: byte 2 = R1 L1 Start Select R2 L2 F1 F3,
    # byte 3 = A B X Y Up Right Down Left
    raw = [0] * 40
    raw[2] = 1 << 5                 # L2
    raw[3] = 1 << 1                 # B
    raw[4:8] = list(struct.pack("<f", 0.7))      # lx: must not matter
    check("L2+B from raw bytes is the e-stop", S.is_estop(S.keys_from_remote_bytes(raw)))
    raw[3] = 1 << 0                 # A instead of B
    check("L2+A is not", not S.is_estop(S.keys_from_remote_bytes(raw)))
    for name in S.BUTTON_BIT:
        k = 1 << S.BUTTON_BIT[name]
        if S.is_estop(k):
            check(f"{name} alone is not the e-stop", False)
    check("all buttons pressed include the e-stop", S.is_estop(0xFFFF))
    check("remote_bytes round trip", S.keys_from_remote_bytes(S.remote_bytes(0xA5C3)) == 0xA5C3)


def test_command_check() -> None:
    print("command check")
    stand = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]
    z, one = [0.0] * 12, [1] * 12
    ok = S.check_command(stand, z, [50] * 12, [3] * 12, z, one)
    check("stand command passes", ok is None, str(ok))
    check("damping passes", S.check_command(z, z, z, [5] * 12, z, one) is None)
    check("WTW gains pass", S.check_command(stand, z, [20] * 12, [0.5] * 12, z, one) is None)
    bad = list(stand)
    bad[4] = float("nan")
    check("NaN caught", "NaN" in (S.check_command(bad, z, [20] * 12, [0.5] * 12, z, one) or ""))
    bad = list(stand)
    bad[2] = 0.5
    check("calf target beyond range caught", "range" in (S.check_command(bad, z, [20] * 12, [0.5] * 12, z, one) or ""))
    check("out-of-range target ignored when kp=0", S.check_command(bad, z, z, [5] * 12, z, one) is None)
    check("kp 200 caught", S.check_command(stand, z, [200] * 12, [1] * 12, z, one) is not None)
    check("negative kd caught", S.check_command(stand, z, [20] * 12, [-1] * 12, z, one) is not None)
    check("tau 30 caught", S.check_command(stand, z, [20] * 12, [1] * 12, [30] * 12, one) is not None)
    check("mode 0 caught", S.check_command(stand, z, [20] * 12, [1] * 12, z, [0] * 12) is not None)
    lie = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65, -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]
    check("lying check: Unitree lie-down pose is lying", S.is_lying(lie))
    check("lying check: stand pose is not", not S.is_lying(stand))
    one_up = list(lie)
    one_up[8] = -1.5
    check("lying check: one leg extended is not", not S.is_lying(one_up))


def test_tilt() -> None:
    print("tilt from quaternion vs MuJoCo")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        r, p, y = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-3, 3)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([r, p, y]), "XYZ")      # MuJoCo: upper case = extrinsic; X,Y,Z extrinsic = roll-pitch-yaw
        rr, pp = S.tilt_from_quat(quat)
        worst = max(worst, abs(rr - r), abs(pp - p))
    check("roll/pitch match", worst < 1e-6, f"worst {worst:.1e} rad")


def test_supervisor_logic() -> None:
    print("supervisor timing logic")
    up = [1.0, 0, 0, 0]
    nokey = [0] * 40

    def fed(t_end, dt=0.002, state=True, cmd=True, remote=True, tick=True, t0=0.0, sup=None):
        s = sup or S.Supervisor(now=t0)
        t = t0
        while t < t_end:
            if state:
                s.on_state(t, True, int(t * 1000) if tick else 7, up, nokey)
            if cmd:
                s.on_command(t, True, None)
            if remote and int(t / dt) % 10 == 0:
                s.on_remote(t, 0)
            if s.check(t):
                return s
            t += dt
        return s

    s = fed(5.0)
    check("nominal 5 s: no trip", not s.tripped, str(s.trip_reason))
    s = S.Supervisor(now=0)
    s.check(59.0)
    check("not armed: no trip before startup timeout", not s.tripped)
    s.check(61.0)
    check("not armed: startup timeout trips", s.tripped and "within" in s.trip_reason)
    s = fed(1.0)
    s = fed(2.0, cmd=False, t0=1.0, sup=s)
    check("controller silent -> trip ~50 ms", s.tripped and "controller silent" in s.trip_reason
          and 0.05 <= s.trip_t - 1.0 <= 0.056, f"{s.trip_reason} at +{(s.trip_t - 1.0) * 1000:.0f} ms")
    s = fed(1.0)
    s = fed(2.0, state=False, t0=1.0, sup=s)
    check("state lost -> trip ~100 ms", s.tripped and "lowstate" in s.trip_reason
          and 0.1 <= s.trip_t - 1.0 <= 0.106, f"{s.trip_reason}")
    s = fed(1.0)
    s = fed(2.0, tick=False, t0=1.0, sup=s)
    check("tick stuck -> trip", s.tripped and "tick" in s.trip_reason, f"{s.trip_reason}")
    s = S.Supervisor(now=0)
    for k in range(500):          # tick never moves from the start (e.g. firmware leaves it 0): not armed
        s.on_state(k * 0.002, True, 0, up, nokey)
        s.on_command(k * 0.002, True, None)
        s.check(k * 0.002)
    check("tick never moving: check not armed (no false trip)", not s.tripped, str(s.trip_reason))
    s = fed(1.0)
    s = fed(2.0, remote=False, t0=1.0, sup=s)
    check("remote silent -> trip ~0.5 s", s.tripped and "remote" in s.trip_reason
          and 0.47 <= s.trip_t - 1.0 <= 0.53, f"{s.trip_reason}")
    s = fed(2.0, remote=False)
    check("remote never seen: no trip", not s.tripped, str(s.trip_reason))
    s = S.Supervisor(S.Limits(remote_check=False), now=0)
    s = fed(1.0, sup=s)
    s = fed(2.0, remote=False, t0=1.0, sup=s)
    check("remote check off: no trip", not s.tripped)
    s = fed(1.0)
    s.on_state(1.0, False, 99999, up, S.remote_bytes(S.ESTOP_KEYS))
    check("bad-CRC state is ignored (even with e-stop bits)", not s.tripped)
    s.on_state(1.0, True, 1000, up, S.remote_bytes(S.ESTOP_KEYS))
    check("e-stop in state trips", s.tripped and "e-stop" in s.trip_reason)
    s = fed(1.0)
    s.on_command(1.0, True, "NaN/inf in command")
    check("bad command trips", s.tripped and "bad command" in s.trip_reason)
    reason = s.trip_reason
    fed(2.0, t0=1.0, sup=s)
    s.on_remote(1.5, S.ESTOP_KEYS)
    check("trip is latched (first reason kept)", s.trip_reason == reason)
    s = fed(1.0)
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0.85, 0, 0]), "XYZ")
    s.on_state(1.0, True, 1000, q, nokey)
    check("roll 0.85 rad trips", s.tripped and "tilt" in s.trip_reason)


def test_safe_dodge_never_trips(n: int = 20) -> None:
    print(f"false alarms: SAFE dodge, {n} harsh-physics runs, every command and tilt sample")
    import show_dodges as sd
    dodge = sd.safe_dodge()
    pol = g.WTWPolicy()
    rng = np.random.default_rng(7)
    worst_tilt, n_cmds, problems = 0.0, 0, []
    for _ in range(n):
        phys = g.harsh_physics(rng)
        d = replace(dodge, pre=dodge.pre + phys.pre_offset)
        sim = g.Go2Sim(phys)
        ctl = g.DodgeController(d, sim, pol)
        while sim.t < d.total - 1e-9:
            qt, kp, kd, ph = ctl.targets(sim.t)
            if phys.target_margin > 0:        # what goes on the wire: the deploy clamp is applied first
                qt = np.clip(qt, sim.q_lo + phys.target_margin, sim.q_hi - phys.target_margin)
            p = S.check_command(qt[POLICY_TO_SDK], [0] * 12, [kp] * 12, [kd] * 12, [0] * 12, [1] * 12)
            n_cmds += 1
            if p:
                problems.append(p)
            sim.step(qt, kp, kd)
            quat = sim.d.qpos[3:7]
            worst_tilt = max(worst_tilt, *map(abs, S.tilt_from_quat(quat)))
    check("no SAFE-dodge command rejected", not problems, f"{n_cmds} commands, {len(problems)} rejected "
          + (problems[0] if problems else ""))
    check("tilt stays below the 0.8 rad trip", worst_tilt < 0.8 * 0.75,
          f"worst {worst_tilt:.2f} rad = {math.degrees(worst_tilt):.0f} deg")


def test_still_dodge_never_trips(n: int = 20) -> None:
    print(f"false alarms: STILL dodge, {n} harsh-physics runs per side (stand + dodge), every command and tilt sample")
    import still_dodge as sd
    pol = g.WTWPolicy()
    rng = np.random.default_rng(7)
    worst = {"tilt": 0.0, "n": 0}
    problems: list[str] = []

    def on_cmd(sim, qt, kp, kd):
        if sim.phys.target_margin > 0:        # what goes on the wire: the deploy clamp is applied first
            qt = np.clip(qt, sim.q_lo + sim.phys.target_margin, sim.q_hi - sim.phys.target_margin)
        p = S.check_command(np.asarray(qt)[POLICY_TO_SDK], [0] * 12, [kp] * 12, [kd] * 12, [0] * 12, [1] * 12)
        worst["n"] += 1
        if p:
            problems.append(p)
        worst["tilt"] = max(worst["tilt"], *map(abs, S.tilt_from_quat(sim.d.qpos[3:7])))

    for side in ("left", "right"):
        d = sd.load(side=side)
        for _ in range(n):
            sd.run(d, g.harsh_physics(rng), pol, on_cmd=on_cmd)
    check("no STILL-dodge command rejected", not problems, f"{worst['n']} commands, {len(problems)} rejected "
          + (problems[0] if problems else ""))
    check("tilt stays below the 0.8 rad trip", worst["tilt"] < 0.8 * 0.75,
          f"worst {worst['tilt']:.2f} rad = {math.degrees(worst['tilt']):.0f} deg")


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_remote()
    test_command_check()
    test_tilt()
    test_supervisor_logic()
    test_safe_dodge_never_trips()
    test_still_dodge_never_trips()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
