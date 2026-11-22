"""STILL dodge: the Go2 stands completely still (PD hold, no stepping) until the trigger, then sidesteps.

Why: the SAFE dodge (go2_dodge_sim.Dodge, results/safe_dodge.json) trots in place for ~2 s before the
kick. On the real dog that trotting drifted ~40 cm and kept drifting, and the throw sensors need a
still dog. Here the policy only starts at the trigger, so there is no lead-in at all:
trigger -> kick + hold sideways -> brake -> ramp back to the stand pose -> still.

The policy needs a 30-step observation history. While the dog stands, `hold()` keeps feeding it
real observations of the stand (zero command, zero action, gait clock parked at `phase0`), so at the
trigger the history says "standing still at the default pose" instead of zeros. The clock then
starts at `phase0`, which picks the diagonal pair that lifts first. The right dodge is the mirror
image: sideways commands negated and the clock shifted by half a cycle (that swaps the two diagonal
pairs, i.e. mirrors the legs).

The same class runs in sim (run() below) and on the robot (deploy/dodge_controller.py --still), so
the validated sequence is the deployed one.

  python still_dodge.py                         # nominal run, left and right
  python optimize_still.py                      # search + harsh validation -> results/still_dodge.json
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

import go2_dodge_sim as g

HERE = Path(__file__).resolve().parent
STILL_PATH = HERE / "results/still_dodge.json"


@dataclass
class StillDodge:
    name: str = "STILL"
    kick_vy: float = 1.4          # +left [m/s]
    kick_s: float = 0.35
    hold_vy: float = 0.5
    dodge_s: float = 0.5          # total sideways command time (>= kick_s)
    brake_vy: float = -0.4
    brake_s: float = 0.05
    brake: float = 1.0            # zero command (incl. brake_s) before the stand ramp
    freq: float = 3.7
    stance_width: float = 0.33
    footswing: float = 0.08
    height: float = -0.03
    phase0: float = 0.0           # gait clock at the trigger (left dodge; right = +0.5)
    prime: bool = True            # feed stand observations into the history before the trigger
    stand: float = 1.5
    stand_kp: float = 40.0
    stand_kd: float = 1.0
    stand_ramp: float = 0.4
    side: str = "left"

    @property
    def sign(self) -> float:
        return -1.0 if self.side == "right" else 1.0

    @property
    def t0(self) -> float:
        """Time from the command to the kick (the SAFE dodge trots ~2.5 s first; here: none)."""
        return 0.0

    @property
    def cmd(self) -> g.Command:
        return self.command(self.kick_vy)

    @property
    def total(self) -> float:
        """Time from the trigger to the end of the sequence (dog standing still again)."""
        return max(self.dodge_s, self.kick_s) + self.brake + self.stand

    def mirrored(self, side: str) -> "StillDodge":
        return replace(self, side=side)

    def scaled(self, s: float) -> "StillDodge":
        """All sideways speed commands x s (the gentle knob)."""
        return replace(self, kick_vy=self.kick_vy * s, hold_vy=self.hold_vy * s, brake_vy=self.brake_vy * s)

    def command(self, vy: float) -> g.Command:
        return g.Command(vy=self.sign * vy, freq=self.freq, stance_width=self.stance_width,
                         footswing=self.footswing, height=self.height)

    @classmethod
    def from_params(cls, p: dict, **kw) -> "StillDodge":
        f = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in p.items() if k in f}, **kw)

    def params(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in ("name", "side")}


def load(gentle: float | None = None, side: str = "left", path: Path = STILL_PATH) -> StillDodge:
    p = json.loads(path.read_text())["chosen"]["params"]
    d = StillDodge.from_params(p, name="STILL" + (f" x{gentle:g}" if gentle else ""))
    if gentle:
        d = d.scaled(gentle)
    return d.mirrored(side)


class StillDodgeController:
    """Phase logic for one robot. Before the trigger call hold() every policy period (50 Hz); after
    start(), call targets(t) with t = time since the trigger, until t >= dodge.total."""

    def __init__(self, dodge: StillDodge, robot, policy: g.WTWPolicy):
        self.d, self.robot, self.pol = dodge, robot, policy
        self.pol.reset()
        self.started = False
        self.stand_from = None

    def _phase0(self) -> float:
        return (self.d.phase0 + (0.5 if self.d.side == "right" else 0.0)) % 1.0

    def hold(self) -> tuple[np.ndarray, float, float]:
        """Stand still at the default pose; keep the policy history filled with the stand."""
        if self.d.prime and not self.started:
            grav, q, dq = self.robot.observe()
            self.pol.gait_index = self._phase0()
            phase, offset, bound = g.GAITS["trot"]
            gi = self.pol.gait_index
            self.pol.clock = np.sin(2 * np.pi * np.array([gi + phase + offset + bound, gi + offset, gi + bound, gi + phase]))
            c = replace(self.d.command(0.0), vy=0.0).vector()
            obs = np.concatenate([grav, c * g.CMD_SCALE, q - g.DEFAULT_Q, dq * g.DOF_VEL_SCALE,
                                  np.zeros(12), np.zeros(12), self.pol.clock]).astype(np.float32)
            self.pol.hist = np.concatenate([self.pol.hist[g.N_OBS:], obs])
        return g.DEFAULT_Q, self.d.stand_kp, self.d.stand_kd

    def start(self) -> None:
        if not self.d.prime:
            self.pol.reset()
        self.pol.gait_index = self._phase0() - g.POLICY_DT * self.d.freq   # first step lands on phase0
        self.started = True

    def targets(self, t: float) -> tuple[np.ndarray, float, float, str]:
        d = self.d
        t1 = max(d.dodge_s, d.kick_s)
        t2 = t1 + d.brake
        if t < t1:
            vy = d.kick_vy if t < d.kick_s else d.hold_vy
            return self.pol.targets(*self.robot.observe(), d.command(vy)), g.KP, g.KD, "dodge"
        if t < t2:
            vy = d.brake_vy if (t - t1 < d.brake_s and d.brake_s > 0.02) else 0.0
            return self.pol.targets(*self.robot.observe(), d.command(vy)), g.KP, g.KD, "brake"
        if self.stand_from is None:
            self.stand_from = self.robot.q
        a = min(1.0, (t - t2) / d.stand_ramp)
        return (1 - a) * self.stand_from + a * g.DEFAULT_Q, d.stand_kp, d.stand_kd, "stand"


class _Shim:
    """What go2_dodge_sim.metrics() reads from res['dodge']."""

    def __init__(self, d: StillDodge):
        self.cmd = g.Command(vy=d.sign)
        self.name = d.name


LOG_KEYS = ("t", "y", "vy", "x", "z", "roll", "pitch", "yaw", "contacts", "phase")


def run(d: StillDodge, phys: g.Physics = g.Physics(), policy: g.WTWPolicy | None = None,
        loads: bool = False, settle: float = 1.5, on_cmd=None) -> dict:
    """One still dodge in sim: `settle` s PD stand (plus phys.pre_offset, which here only varies how
    long it stood), trigger, dodge, brake, stand. Log t is time since the trigger."""
    sim = g.Go2Sim(phys)
    ctl = StillDodgeController(d, sim, policy or g.WTWPolicy())
    meter = g.LoadMeter(sim) if loads else None
    settle = max(0.6, settle + phys.pre_offset)
    L = {k: [] for k in LOG_KEYS}
    origin = heading = None
    fell = False
    hold_xy = []
    n_hold = round(settle / g.POLICY_DT)
    n_all = n_hold + round(d.total / g.POLICY_DT)
    for k in range(n_all):
        t = (k - n_hold) * g.POLICY_DT                        # time since the trigger
        if k < n_hold:
            q, kp, kd = ctl.hold()
            phase = "hold"
        else:
            if k == n_hold:
                ctl.start()
                origin, heading = sim.pos, sim.rpy()[2]
            q, kp, kd, phase = ctl.targets(t)
        if on_cmd is not None:                                 # tests: every command as sent
            on_cmd(sim, q, kp, kd)
        if meter is None or k < n_hold - 10:
            sim.step(q, kp, kd)
        else:
            for _ in range(sim.substeps):
                sim.apply_pd(q, kp, kd)
                mujoco.mj_step(sim.m, sim.d)
                meter.sample()
            sim.t += g.POLICY_DT
        rpy = sim.rpy()
        if k >= n_hold - 25 and k < n_hold:                  # last 0.5 s before the trigger
            hold_xy.append((sim.pos[:2].copy(), np.abs(sim.dq).max(), np.abs(sim.vel).max()))
        if origin is not None:
            c, s = math.cos(-heading), math.sin(-heading)
            dp, v = sim.pos - origin, sim.vel
            x, y, vy = c * dp[0] - s * dp[1], s * dp[0] + c * dp[1], s * v[0] + c * v[1]
        else:
            x = y = vy = 0.0
        for key, val in zip(LOG_KEYS, (t + g.POLICY_DT, y, vy, x, sim.pos[2], rpy[0], rpy[1],
                                       rpy[2] - (heading or 0.0), int(sim.foot_contacts().sum()), phase)):
            L[key].append(val)
        fell |= sim.pos[2] < 0.15 or abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8 or sim.trunk_on_floor()
    res = {"dodge": _Shim(d), "phys": phys, "log": {k: np.array(v) for k, v in L.items()}, "fell": bool(fell)}
    if meter is not None:
        res["loads"] = meter.summary()
    xy = np.array([h[0] for h in hold_xy])
    res["hold"] = {"drift_m": float(np.linalg.norm(xy - xy[0], axis=1).max()),
                   "joint_speed_max": float(max(h[1] for h in hold_xy)),
                   "trunk_speed_max": float(max(h[2] for h in hold_xy))}
    return res


def metrics(res: dict) -> dict:
    m = g.metrics(res)
    L = res["log"]
    after = L["t"] > 0
    m["hold_drift_m"] = res["hold"]["drift_m"]
    m["hold_trunk_speed"] = res["hold"]["trunk_speed_max"]
    m["y_max"] = float(np.max(np.sign(res["dodge"].cmd.vy) * L["y"][after]))
    m["x_abs_max"] = float(np.abs(L["x"][after]).max())
    # still at the end: trunk travel over the last 0.5 s
    last = L["t"] >= L["t"][-1] - 0.5
    m["end_travel_m"] = float(np.hypot(np.ptp(L["x"][last]), np.ptp(L["y"][last])))
    return m


def main() -> None:
    import argparse
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gentle", type=float, default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    pol = g.WTWPolicy()
    base = load(a.gentle) if STILL_PATH.exists() else StillDodge()
    for side in ("left", "right"):
        d = base.mirrored(side)
        m = metrics(run(d, g.Physics(), pol, loads=True))
        print(f"{d.name + ' ' + d.side:22s} t10 {m['t_10cm']:.2f} t20 {m['t_20cm']:.2f} t30 {m['t_30cm']:.2f}  final {m['y_final']:.2f} m "
              f"(max {m['y_max']:.2f}), x {m['x_drift']:+.2f}, yaw {m['yaw_drift_deg']:+.0f} deg, tilt {m['peak_tilt_deg']:.1f} deg, "
              f"foot {m['foot_force_bw']:.2f} BW, acc {m['trunk_acc_g']:.2f} g, margin {m['limit_margin']:.2f}, "
              f"hold drift {m['hold_drift_m'] * 1000:.1f} mm, {'stable' if m['stable_end'] else 'NOT STABLE'}")


if __name__ == "__main__":
    main()
