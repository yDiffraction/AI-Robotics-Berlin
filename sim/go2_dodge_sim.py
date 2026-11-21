"""Sideways dodge for the Unitree Go2 in MuJoCo, driven by the public walk-these-ways Go2 policy.

Starting point: https://github.com/Teddy-Liao/walk-these-ways-go2 (pretrained Go2 policy,
gait-conditioned: you command vx/vy/yaw plus step frequency, gait type, footswing height ...).
Robot model: unitreerobotics/unitree_mujoco (go2.xml, Unitree's official MJCF).

The observation, action scaling, PD gains and gait clock follow the repo's own Go2
deployment code (go2_gym_deploy/envs/lcm_agent.py, lcm_position_go2.cpp: kp=20, kd=0.5),
so what runs here is what that repo sends to the real dog.

Scenario (all times in seconds):
  settle  : PD hold of the default pose
  pre     : policy running with zero velocity (it trots in place)   [--from-stand: PD hold instead]
  dodge   : policy with lateral command vy                          <- trigger at t0
  brake   : policy with zero velocity
  stand   : PD hold of the default pose, ramped in from the current joint angles

Run inside WSL (see sim/README.md):
  python sim/go2_dodge_sim.py --sweep                 # compare dodge variants, print a table
  python sim/go2_dodge_sim.py --vy 0.6 --freq 4
  python sim/show_dodges.py                           # 3D viewer, dodges side by side
  python sim/go2_dodge_sim.py --sweep --rand 20       # robustness over randomized physics
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np
import torch

# WSL: full install in /root/go2_sim; Windows: model + policy copied to ~/go2_sim (scripts/install_go2_sim_windows.ps1)
GO2_SIM = Path(os.environ.get("GO2_SIM", Path.home() / "go2_sim" if os.name == "nt" else "/root/go2_sim"))
WTW_CKPT = GO2_SIM / "walk-these-ways-go2/runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints"
SCENE = GO2_SIM / "unitree_mujoco/unitree_robots/go2/scene.xml"

# ---- policy constants (walk-these-ways-go2 training config + deploy code) -------------------
LEGS = ("FL", "FR", "RL", "RR")                       # policy joint order
JOINTS = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]
DEFAULT_Q = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
KP, KD = 20.0, 0.5                                    # deploy gains on the real Go2
ACTION_SCALE, HIP_SCALE, CLIP_ACT = 0.25, 0.5, 10.0
POLICY_DT = 0.02                                      # 50 Hz
N_OBS, N_HIST = 70, 30
# lin_vel, lin_vel, ang_vel, body_height, freq, phase, offset, bound, duration,
# footswing, pitch, roll, stance_width, stance_length, aux
CMD_SCALE = np.array([2, 2, 0.25, 2, 1, 1, 1, 1, 1, 0.15, 0.3, 0.3, 1, 1, 1], dtype=np.float32)
DOF_VEL_SCALE = 0.05
# (phase, offset, bound) as in walk-these-ways; foot clocks: FL=g+p+o+b, FR=g+o, RL=g+b, RR=g+p
GAITS = {"trot": (0.5, 0.0, 0.0), "pronk": (0.0, 0.0, 0.0), "pace": (0.0, 0.5, 0.0), "bound": (0.0, 0.0, 0.5)}
TRAINED_VY = 0.6                                      # |vy| range seen in training

# Go2 joint motors (GO-M8010-6): 23.7 Nm / ~30 rad/s; knee has ~1.9x extra reduction -> 45.4 Nm / ~15.7 rad/s
MAX_SPEED = np.array([30.1, 30.1, 15.7] * 4)


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    height: float = 0.0
    freq: float = 3.0           # steps per second, trained range 2..4
    gait: str = "trot"
    duration: float = 0.5
    footswing: float = 0.08     # trained range 0.03..0.35
    pitch: float = 0.0
    stance_width: float = 0.33  # trained range 0.10..0.45
    stance_length: float = 0.40

    def vector(self) -> np.ndarray:
        phase, offset, bound = GAITS[self.gait]
        return np.array([self.vx, self.vy, self.yaw, self.height, self.freq, phase, offset, bound,
                         self.duration, self.footswing, self.pitch, 0.0, self.stance_width,
                         self.stance_length, 0.0], dtype=np.float32)


class WTWPolicy:
    """walk-these-ways student policy: adaptation module + body, on a 30-step observation history."""

    def __init__(self, ckpt: Path = WTW_CKPT):
        self.body = torch.jit.load(str(ckpt / "body_latest.jit"), map_location="cpu").eval()
        self.adapt = torch.jit.load(str(ckpt / "adaptation_module_latest.jit"), map_location="cpu").eval()
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(N_OBS * N_HIST, np.float32)
        self.action = np.zeros(12, np.float32)
        self.last_action = np.zeros(12, np.float32)
        self.gait_index = 0.0
        self.clock = np.zeros(4, np.float32)

    def targets(self, gravity: np.ndarray, q: np.ndarray, dq: np.ndarray, cmd: Command) -> np.ndarray:
        c = cmd.vector()
        # gait clock first, as in training (_step_contact_targets runs before compute_observations)
        phase, offset, bound = c[5], c[6], c[7]
        self.gait_index = (self.gait_index + POLICY_DT * c[4]) % 1.0
        g = self.gait_index
        self.clock = np.sin(2 * np.pi * np.array([g + phase + offset + bound, g + offset, g + bound, g + phase]))

        obs = np.concatenate([gravity, c * CMD_SCALE, q - DEFAULT_Q, dq * DOF_VEL_SCALE,
                              self.action, self.last_action, self.clock]).astype(np.float32)
        self.hist = np.concatenate([self.hist[N_OBS:], obs])
        with torch.no_grad():
            h = torch.from_numpy(self.hist)[None]
            a = self.body(torch.cat([h, self.adapt(h)], -1))[0].numpy()
        self.last_action = self.action
        self.action = np.clip(a, -CLIP_ACT, CLIP_ACT).astype(np.float32)
        t = self.action * ACTION_SCALE
        t[[0, 3, 6, 9]] *= HIP_SCALE
        return t + DEFAULT_Q


@dataclass
class Physics:
    friction: float = 0.7          # foot-floor sliding friction (the MJCF ships 0.4 on the feet)
    mass_add: float = 0.0          # kg added to the trunk (payload, e.g. Kria + battery)
    motor_strength: float = 1.0    # scales the torque limits
    latency: float = 0.0           # s, delay between joint target and the motor seeing it
    torque_speed: bool = True      # torque falls off above half of max joint speed
    timestep: float = 0.002
    obs_noise: float = 0.0         # 1.0 = legged_gym noise: joint pos 0.01 rad, joint vel 1.5 rad/s, gravity 0.05
    target_margin: float = 0.0     # clamp joint targets this far inside the joint range [rad] (deploy safety)
    pre_offset: float = 0.0        # shifts the trigger time -> random gait phase at the trigger
    seed: int = 0                  # for the observation noise


def flat_scene_spec(scene: Path = SCENE) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(scene))
    for geom in list(spec.worldbody.geoms):      # flat floor only (drop the demo steps/boxes)
        if geom.name != "floor":
            spec.delete(geom)
    return spec


class Go2Robot:
    """One Go2 inside a (possibly multi-robot) MuJoCo model; names carry `prefix`."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, phys: Physics, prefix: str = "", home_xy=(0.0, 0.0)):
        self.m, self.d, self.phys, self.prefix, self.home_xy = m, d, phys, prefix, home_xy
        p = prefix
        self.qadr = np.array([m.jnt_qposadr[m.joint(p + n).id] for n in JOINTS])
        self.vadr = np.array([m.jnt_dofadr[m.joint(p + n).id] for n in JOINTS])
        self.act = np.array([m.actuator(p + n.replace("_joint", "")).id for n in JOINTS])
        self.tau_max = m.actuator_ctrlrange[self.act, 1] * phys.motor_strength
        self.trunk = m.body(p + "base_link").id
        free = m.body_jntadr[self.trunk]
        self.bq, self.bv = m.jnt_qposadr[free], m.jnt_dofadr[free]
        m.body_mass[self.trunk] += phys.mass_add
        self.floor = m.geom("floor").id
        self.feet = np.array([m.geom(p + leg).id for leg in LEGS])
        for gid in (*self.feet, self.floor):
            m.geom_friction[gid, 0] = phys.friction
        self.delay = deque(maxlen=max(1, round(phys.latency / m.opt.timestep) + 1))
        jr = m.jnt_range[[m.joint(p + n).id for n in JOINTS]]
        self.q_lo, self.q_hi = jr[:, 0].copy(), jr[:, 1].copy()
        self.rng = np.random.default_rng(phys.seed)

    def reset(self) -> None:
        self.d.qpos[self.bq:self.bq + 3] = [*self.home_xy, 0.30]
        self.d.qpos[self.bq + 3:self.bq + 7] = [1, 0, 0, 0]
        self.d.qvel[self.bv:self.bv + 6] = 0
        self.d.qpos[self.qadr] = DEFAULT_Q
        self.d.qvel[self.vadr] = 0
        self.delay.clear()

    # ---- state -------------------------------------------------------------------------------
    @property
    def q(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.d.qvel[self.vadr].copy()

    @property
    def pos(self) -> np.ndarray:
        return self.d.qpos[self.bq:self.bq + 3].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.d.qvel[self.bv:self.bv + 3].copy()

    def rot(self) -> np.ndarray:
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, self.d.qpos[self.bq + 3:self.bq + 7])
        return r.reshape(3, 3)

    def gravity(self) -> np.ndarray:
        return self.rot().T @ np.array([0.0, 0.0, -1.0])

    def observe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """What the policy sees: gravity, q, dq, with sensor noise if phys.obs_noise > 0."""
        grav, q, dq = self.gravity(), self.q, self.dq
        k = self.phys.obs_noise
        if k > 0:
            grav = grav + self.rng.uniform(-1, 1, 3) * 0.05 * k
            q = q + self.rng.uniform(-1, 1, 12) * 0.01 * k
            dq = dq + self.rng.uniform(-1, 1, 12) * 1.5 * k
        return grav, q, dq

    def rpy(self) -> np.ndarray:
        r = self.rot()
        return np.array([math.atan2(r[2, 1], r[2, 2]), -math.asin(np.clip(r[2, 0], -1, 1)), math.atan2(r[1, 0], r[0, 0])])

    def foot_contacts(self) -> np.ndarray:
        c = np.zeros(4, bool)
        for k in range(self.d.ncon):
            g = {self.d.contact[k].geom1, self.d.contact[k].geom2}
            if self.floor in g:
                c |= np.isin(self.feet, list(g))
        return c

    def trunk_on_floor(self) -> bool:
        for k in range(self.d.ncon):
            g1, g2 = self.d.contact[k].geom1, self.d.contact[k].geom2
            if self.floor in (g1, g2):
                other = g2 if g1 == self.floor else g1
                if other not in self.feet and self.m.geom_bodyid[other] == self.trunk:
                    return True
        return False

    # ---- control -----------------------------------------------------------------------------
    def apply_pd(self, q_target: np.ndarray, kp: float, kd: float) -> None:
        """Joint PD for one sim step (the motor driver does this on the dog)."""
        if self.phys.target_margin > 0:
            q_target = np.clip(q_target, self.q_lo + self.phys.target_margin, self.q_hi - self.phys.target_margin)
        self.delay.append(q_target)
        tgt = self.delay[0]
        q, dq = self.q, self.dq
        tau = kp * (tgt - q) - kd * dq
        lim = self.tau_max.copy()
        if self.phys.torque_speed:   # crude torque-speed curve, only while driving the motion
            frac = np.clip((np.abs(dq) / MAX_SPEED - 0.5) / 0.5, 0, 1)
            lim = np.where(tau * dq > 0, lim * (1 - frac), lim)
        self.d.ctrl[self.act] = np.clip(tau, -lim, lim)


class Go2Sim(Go2Robot):
    """Single Go2 on a flat floor."""

    def __init__(self, phys: Physics = Physics(), scene: Path = SCENE):
        m = flat_scene_spec(scene).compile()
        m.opt.timestep = phys.timestep
        super().__init__(m, mujoco.MjData(m), phys)
        self.substeps = round(POLICY_DT / phys.timestep)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        super().reset()
        mujoco.mj_forward(self.m, self.d)
        self.t = 0.0

    def step(self, q_target: np.ndarray, kp: float = KP, kd: float = KD) -> None:
        for _ in range(self.substeps):
            self.apply_pd(q_target, kp, kd)
            mujoco.mj_step(self.m, self.d)
        self.t += POLICY_DT


@dataclass
class Dodge:
    name: str = "trot 3Hz vy0.6"
    cmd: Command = field(default_factory=lambda: Command(vy=TRAINED_VY))
    from_stand: bool = False      # dog stands still (PD hold) until the trigger
    kick_vy: float | None = None  # optional over-command for the first kick_s seconds of the dodge
    kick_s: float = 0.0
    brake_vy: float | None = None  # optional counter-command for the first brake_s seconds of the brake
    brake_s: float = 0.0
    pre: float = 2.0
    dodge: float = 1.0
    brake: float = 1.0
    stand: float = 1.5
    stand_kp: float = 40.0
    stand_kd: float = 1.0
    stand_ramp: float = 0.4
    settle: float = 0.5

    @property
    def t0(self) -> float:
        return self.settle + self.pre

    @property
    def total(self) -> float:
        return self.t0 + self.dodge + self.brake + self.stand


class DodgeController:
    """Phase logic of one dodge for one robot, plus logging; call step() once per policy period."""

    LOG_KEYS = ("t", "y", "vy", "x", "z", "roll", "pitch", "yaw", "contacts", "phase")

    def __init__(self, dodge: Dodge, robot: Go2Robot, policy: WTWPolicy):
        self.dodge, self.robot, self.pol = dodge, robot, policy
        self.reset()

    def reset(self) -> None:
        self.pol.reset()
        self.log = {k: [] for k in self.LOG_KEYS}
        self.origin = self.heading = None
        self.stand_from = None
        self.fell = False

    def targets(self, t: float) -> tuple[np.ndarray, float, float, str]:
        d, r = self.dodge, self.robot
        t0 = d.t0
        t1, t2 = t0 + d.dodge, t0 + d.dodge + d.brake
        if t < d.settle or (d.from_stand and t < t0):
            return DEFAULT_Q, d.stand_kp, d.stand_kd, "hold"
        if t < t2:
            phase = "dodge" if t0 <= t < t1 else ("brake" if t >= t1 else "pre")
            cmd = d.cmd if phase == "dodge" else replace(d.cmd, vx=0.0, vy=0.0, yaw=0.0)
            if phase == "dodge" and d.kick_vy is not None and t - t0 < d.kick_s:
                cmd = replace(cmd, vy=d.kick_vy)
            if phase == "brake" and d.brake_vy is not None and t - t1 < d.brake_s:
                cmd = replace(cmd, vy=d.brake_vy)
            return self.pol.targets(*r.observe(), cmd), KP, KD, phase
        if self.stand_from is None:
            self.stand_from = r.q
        a = min(1.0, (t - t2) / d.stand_ramp)
        return (1 - a) * self.stand_from + a * DEFAULT_Q, d.stand_kp, d.stand_kd, "stand"

    def record(self, t_after: float, phase: str) -> None:
        r, L = self.robot, self.log
        if self.origin is None and t_after >= self.dodge.t0 - 1e-9:
            self.origin, self.heading = r.pos, r.rpy()[2]
        rpy = r.rpy()
        if self.origin is not None:
            c, s = math.cos(-self.heading), math.sin(-self.heading)
            dp, v = r.pos - self.origin, r.vel
            x, y = c * dp[0] - s * dp[1], s * dp[0] + c * dp[1]
            vy = s * v[0] + c * v[1]
        else:
            x = y = vy = 0.0
        for k, val in zip(self.LOG_KEYS, (t_after - self.dodge.t0, y, vy, x, r.pos[2], rpy[0], rpy[1],
                                          rpy[2] - (self.heading or 0.0), int(r.foot_contacts().sum()), phase)):
            L[k].append(val)
        self.fell |= r.pos[2] < 0.15 or abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8 or r.trunk_on_floor()

    def result(self, phys: Physics) -> dict:
        return {"dodge": self.dodge, "phys": phys, "log": {k: np.array(v) for k, v in self.log.items()},
                "fell": bool(self.fell)}


class LoadMeter:
    """Mechanical load per sim step: foot forces, touchdown speed, torque, joint speed/limits, trunk accel."""

    def __init__(self, robot: Go2Robot):
        r, m = robot, robot.m
        self.r = r
        self.weight = 9.81 * float(sum(m.body_mass[b] for b in range(m.nbody)
                                       if b == r.trunk or _below(m, b, r.trunk)))
        imu = m.site(r.prefix + "imu").id
        sid = [k for k in range(m.nsensor)
               if m.sensor_type[k] == mujoco.mjtSensor.mjSENS_ACCELEROMETER and m.sensor_objid[k] == imu][0]
        self.acc_adr = m.sensor_adr[sid]
        self.f6, self.v6 = np.zeros(6), np.zeros(6)
        self.in_contact = np.ones(4, bool)
        self.foot_of = {int(gid): j for j, gid in enumerate(r.feet)}
        self.foot, self.tau, self.speed, self.margin, self.acc, self.touch = [], [], [], [], [], []

    def sample(self) -> None:
        r, m, d = self.r, self.r.m, self.r.d
        f = np.zeros(4)
        if d.ncon:
            geoms = d.contact.geom[:d.ncon]                       # vectorised contact filter
            for k in np.nonzero((geoms == r.floor).any(1))[0]:
                g1, g2 = geoms[k]
                j = self.foot_of.get(int(g2 if g1 == r.floor else g1))
                if j is not None:
                    mujoco.mj_contactForce(m, d, int(k), self.f6)
                    f[j] += self.f6[0]
        now = f > 1.0
        if (now & ~self.in_contact).any():
            for j in np.nonzero(now & ~self.in_contact)[0]:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, r.feet[j], self.v6, 0)
                self.touch.append(max(0.0, -self.v6[5]))
        self.in_contact = now
        q, dq = d.qpos[r.qadr], d.qvel[r.vadr]
        self.foot.append(f.max() / self.weight)
        self.tau.append((np.abs(d.actuator_force[r.act]) / r.tau_max).max())
        self.speed.append((np.abs(dq) / MAX_SPEED).max())
        self.margin.append(min((q - r.q_lo).min(), (r.q_hi - q).min()))
        a = d.xmat[r.trunk].reshape(3, 3) @ d.sensordata[self.acc_adr:self.acc_adr + 3]
        a[2] -= 9.81
        self.acc.append(math.sqrt(a @ a) / 9.81)

    def summary(self) -> dict:
        def peak(x, n=5):          # 10 ms moving average drops single-step contact spikes
            x = np.asarray(x)
            return float(np.convolve(x, np.ones(n) / n, mode="same").max()) if len(x) >= n else float(np.max(x))
        tv = np.array(self.touch) if self.touch else np.zeros(1)
        return {"foot_force_bw": peak(self.foot), "touchdown_p95": float(np.percentile(tv, 95)),
                "torque_frac": peak(self.tau), "speed_frac": peak(self.speed),
                "trunk_acc_g": peak(self.acc), "limit_margin": float(np.min(self.margin))}


def _below(m, b, root) -> bool:
    while b > 0:
        b = m.body_parentid[b]
        if b == root:
            return True
    return False


def run(dodge: Dodge, phys: Physics = Physics(), policy: WTWPolicy | None = None, loads: bool = False) -> dict:
    """One dodge. phys.pre_offset shifts the trigger (gait phase); loads=True also measures the
    mechanical load from 0.2 s before the trigger to the end (slower)."""
    if phys.pre_offset:
        dodge = replace(dodge, pre=dodge.pre + phys.pre_offset)
    sim = Go2Sim(phys)
    ctl = DodgeController(dodge, sim, policy or WTWPolicy())
    meter = LoadMeter(sim) if loads else None
    start = dodge.t0 - 0.2
    while sim.t < dodge.total - 1e-9:
        q_target, kp, kd, phase = ctl.targets(sim.t)
        if meter is None or sim.t + POLICY_DT < start:
            sim.step(q_target, kp, kd)
        else:
            for _ in range(sim.substeps):
                sim.apply_pd(q_target, kp, kd)
                mujoco.mj_step(sim.m, sim.d)
                meter.sample()
            sim.t += POLICY_DT
        ctl.record(sim.t, phase)
    res = ctl.result(phys)
    if meter is not None:
        res["loads"] = meter.summary()
    return res


def metrics(res: dict) -> dict:
    L = res["log"]
    t, y = L["t"], L["y"]
    sign = np.sign(res["dodge"].cmd.vy) or 1.0
    ys = sign * y
    after = t >= 0

    def t_reach(d):
        idx = np.nonzero(after & (ys >= d))[0]
        return float(t[idx[0]]) if len(idx) else float("nan")

    def y_at(tt):
        return float(ys[np.argmin(np.abs(t - tt))])

    end = L["phase"] == "stand"
    last = slice(-10, None)   # last 0.2 s
    moving = np.isin(L["phase"], ["dodge", "brake"])
    return {
        "peak_tilt_deg": float(np.degrees(np.maximum(np.abs(L["roll"]), np.abs(L["pitch"]))[moving].max())),
        "min_height": float(L["z"][moving].min()),
        "fell": res["fell"],
        "t_5cm": t_reach(0.05), "t_10cm": t_reach(0.10), "t_20cm": t_reach(0.20), "t_30cm": t_reach(0.30),
        "y_0.3s": y_at(0.3), "y_0.5s": y_at(0.5), "y_1.0s": y_at(1.0),
        "vy_peak": float(np.max(sign * L["vy"][after])),
        "y_final": float(ys[-1]), "x_drift": float(L["x"][-1]), "yaw_drift_deg": float(np.degrees(L["yaw"][-1])),
        "final_roll_deg": float(np.degrees(np.abs(L["roll"][last]).max())),
        "final_pitch_deg": float(np.degrees(np.abs(L["pitch"][last]).max())),
        "final_speed": float(np.abs(np.gradient(L["y"][last], POLICY_DT)).max()),
        "final_feet_down": int(L["contacts"][last].min()),
        **res.get("loads", {}),
        "stable_end": bool(not res["fell"] and L["contacts"][last].min() == 4
                           and np.degrees(np.abs(L["roll"][last]).max()) < 5
                           and np.degrees(np.abs(L["pitch"][last]).max()) < 5
                           and end.any()),
    }


# Hand-tuned with an earlier grid search; sim/optimize_dodge.py finds equally fast, less tilting settings (results/optimize.json).
RECOMMENDED = Dodge("RECOMMENDED: trot 3Hz kick 2.0/0.3s, 1.0, 0.5s", Command(vy=1.0, freq=3.0),
                    kick_vy=2.0, kick_s=0.3, dodge=0.5)


def sweep_variants() -> list[Dodge]:
    v = TRAINED_VY
    return [
        Dodge("trot 3Hz vy0.6 from standstill", Command(vy=v, freq=3.0), from_stand=True),
        Dodge("trot 3Hz vy0.6", Command(vy=v, freq=3.0)),
        Dodge("trot 4Hz vy0.6", Command(vy=v, freq=4.0)),
        Dodge("trot 4Hz vy0.6 wide stance", Command(vy=v, freq=4.0, stance_width=0.40)),
        Dodge("trot 4Hz vy0.6 high swing", Command(vy=v, freq=4.0, footswing=0.12)),
        Dodge("pace 4Hz vy0.6", Command(vy=v, freq=4.0, gait="pace")),
        Dodge("pronk 4Hz vy0.6", Command(vy=v, freq=4.0, gait="pronk")),
        Dodge("bound 4Hz vy0.6", Command(vy=v, freq=4.0, gait="bound")),
        Dodge("trot 4Hz vy1.0 (beyond training)", Command(vy=1.0, freq=4.0)),
        Dodge("trot 4Hz vy1.5 (beyond training)", Command(vy=1.5, freq=4.0)),
        Dodge("trot 4Hz vy2.0 (beyond training)", Command(vy=2.0, freq=4.0)),
        Dodge("trot 4Hz vy2.5 (beyond training)", Command(vy=2.5, freq=4.0)),
        Dodge("trot 4Hz vy3.0 (beyond training)", Command(vy=3.0, freq=4.0)),
        Dodge("trot 3Hz vy2.0", Command(vy=2.0, freq=3.0)),
        Dodge("trot 4Hz kick 3.0 for 0.3s, then 1.0", Command(vy=1.0, freq=4.0), kick_vy=3.0, kick_s=0.3),
        RECOMMENDED,
    ]


def harsh_physics(rng: np.random.Generator) -> Physics:
    """Deployment-like worst cases: sensor noise, joint-target clamp (as the deploy script will do),
    random gait phase at the trigger, payload up to +3 kg, weak motors, up to 30 ms latency."""
    return Physics(friction=rng.uniform(0.35, 1.1), mass_add=rng.uniform(-1.0, 3.0),
                   motor_strength=rng.uniform(0.8, 1.0), latency=rng.uniform(0.0, 0.03),
                   obs_noise=1.0, target_margin=0.1, pre_offset=rng.uniform(-0.3, 0.3),
                   seed=int(rng.integers(1 << 30)))


def random_physics(rng: np.random.Generator, base: Physics) -> Physics:
    return replace(base, friction=rng.uniform(0.4, 1.0), mass_add=rng.uniform(-1.0, 2.0),
                   motor_strength=rng.uniform(0.85, 1.0), latency=rng.uniform(0.0, 0.02))


def fmt(x, nd=2):
    return "  n/a" if isinstance(x, float) and math.isnan(x) else f"{x:5.{nd}f}"


def print_table(rows: list[tuple[str, dict]], rand: int = 0) -> None:
    head = f"{'variant':36s} {'t5cm':>5s} {'t10':>5s} {'t20':>5s} {'t30':>5s} {'y@.3':>5s} {'y@.5':>5s} {'y@1':>5s} {'vpk':>5s} {'yaw°':>5s}  end"
    print(head)
    print("-" * len(head))
    for name, m in rows:
        end = "FELL" if m["fell"] else ("stable" if m["stable_end"] else "unsettled")
        if rand:
            end = f"{m['ok']}/{rand} ok"
        print(f"{name:36s} {fmt(m['t_5cm'])} {fmt(m['t_10cm'])} {fmt(m['t_20cm'])} {fmt(m['t_30cm'])} "
              f"{fmt(m['y_0.3s'])} {fmt(m['y_0.5s'])} {fmt(m['y_1.0s'])} {fmt(m['vy_peak'])} "
              f"{fmt(m['yaw_drift_deg'], 0)}  {end}")


def plot(results: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for r in results:
        L = r["log"]
        sign = np.sign(r["dodge"].cmd.vy) or 1.0
        ax[0].plot(L["t"], sign * L["y"], label=r["dodge"].name + (" (fell)" if r["fell"] else ""))
        ax[1].plot(L["t"], sign * L["vy"])
    d = results[0]["dodge"]
    for a in ax:
        a.axvline(0, color="k", lw=0.8)
        a.axvline(d.dodge, color="k", lw=0.8, ls="--")
        a.axvline(d.dodge + d.brake, color="k", lw=0.8, ls=":")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("sideways displacement [m]")
    ax[1].set_ylabel("sideways velocity [m/s]")
    ax[1].set_xlabel("time since dodge command [s]   (| dodge | brake | stand)")
    ax[0].set_xlim(-0.5, d.dodge + d.brake + d.stand)
    ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"plot -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vy", type=float, default=TRAINED_VY, help="lateral command [m/s], +left; trained |vy|<=0.6")
    ap.add_argument("--freq", type=float, default=3.0, help="step frequency [Hz], trained 2..4")
    ap.add_argument("--gait", choices=list(GAITS), default="trot")
    ap.add_argument("--footswing", type=float, default=0.08)
    ap.add_argument("--stance-width", type=float, default=0.33)
    ap.add_argument("--dodge-s", type=float, default=None, help="how long vy is commanded (default 1.0)")
    ap.add_argument("--kick-vy", type=float, default=None, help="over-command vy for the first --kick-s seconds")
    ap.add_argument("--kick-s", type=float, default=0.3)
    ap.add_argument("--from-stand", action="store_true", help="stand still until the trigger instead of trotting in place")
    ap.add_argument("--friction", type=float, default=Physics.friction)
    ap.add_argument("--mass-add", type=float, default=0.0)
    ap.add_argument("--latency", type=float, default=0.0, help="actuation delay [s]")
    ap.add_argument("--no-torque-speed", action="store_true", help="ideal motors (no torque fall-off with speed)")
    ap.add_argument("--sweep", action="store_true", help="run the preset variants")
    ap.add_argument("--recommended", action="store_true", help="run the tuned dodge (kick 2.0/0.3s, then 1.0 for 0.5s)")
    ap.add_argument("--rand", type=int, default=0, help="N runs per variant with randomized friction/mass/motors/latency")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default="", help="save displacement/velocity plot (png)")
    ap.add_argument("--json", default="", help="save metrics as json")
    args = ap.parse_args()

    phys = Physics(friction=args.friction, mass_add=args.mass_add, latency=args.latency,
                   torque_speed=not args.no_torque_speed)
    if args.sweep:
        variants = sweep_variants()
    elif args.recommended:
        variants = [replace(RECOMMENDED)]
    else:
        cmd = Command(vy=args.vy, freq=args.freq, gait=args.gait, footswing=args.footswing,
                      stance_width=args.stance_width)
        name = (f"{args.gait} {args.freq:g}Hz vy{args.vy:g}" + (" from standstill" if args.from_stand else "")
                + (f" kick{args.kick_vy:g}/{args.kick_s:g}s" if args.kick_vy is not None else ""))
        variants = [Dodge(name, cmd, from_stand=args.from_stand, dodge=args.dodge_s or 1.0,
                          kick_vy=args.kick_vy, kick_s=args.kick_s)]
    if args.dodge_s is not None:
        for v in variants:
            v.dodge = args.dodge_s

    torch.set_num_threads(1)   # tiny MLP: threads only add overhead
    policy = WTWPolicy()
    rows, results, out = [], [], {}
    rng = np.random.default_rng(args.seed)
    for v in variants:
        if args.rand:
            ms = [metrics(run(v, random_physics(rng, phys), policy)) for _ in range(args.rand)]
            agg = {k: float(np.nanmedian([m[k] for m in ms])) for k in ms[0] if k not in ("fell", "stable_end")}
            agg["ok"] = sum(m["stable_end"] for m in ms)
            agg["fell"] = any(m["fell"] for m in ms)
            agg["stable_end"] = agg["ok"] == args.rand
            rows.append((v.name, agg))
            out[v.name] = {"median": agg, "runs": ms}
            continue
        res = run(v, phys, policy)
        m = metrics(res)
        rows.append((v.name, m))
        results.append(res)
        out[v.name] = m

    if args.rand:
        print(f"medians over {args.rand} randomized runs (friction 0.4-1.0, trunk mass -1..+2 kg, "
              f"motor strength 85-100%, latency 0-20 ms)")
    print("times in s after the dodge command, distances in m, velocities in m/s")
    print_table(rows, args.rand)
    if args.plot and results:
        plot(results, args.plot)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str))
        print(f"metrics -> {args.json}")


if __name__ == "__main__":
    main()
