"""Mechanical load of each dodge vs. normal walking: foot impacts, motor torque/speed, accelerations.

Absolute numbers from a rigid-contact simulator are rough (contact stiffness, no foot rubber, no
gear backlash), so every dodge is compared with two references the real Go2 does all the time:
trotting in place and walking sideways at the trained speed (0.6 m/s).

Per run, measured every 2 ms sim step (peaks use a 10 ms moving average to drop single-step
contact spikes; "raw" peaks are reported too):
  foot force      normal contact force per foot, in body weights
  touchdown       foot speed into the ground at first contact
  torque          |tau| / motor limit, and % of time a motor is saturated
  joint speed     |dq| / max motor speed (30.1 rad/s hip/thigh, 15.7 rad/s knee)
  joint accel     |ddq| (rad/s^2)
  trunk accel     IMU accelerometer minus gravity, in g
  limit margin    smallest distance of any joint to its mechanical range limit (rad)

  cd sim && python load_check.py            # nominal + 10 randomized physics, ~1 min
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import mujoco
import numpy as np
import torch

import go2_dodge_sim as g
import show_dodges as sd

G = 9.81


def references_and_dodges() -> list[g.Dodge]:
    still = g.Dodge("trot in place (reference)", g.Command(vy=0.0))
    trained = g.Dodge("walk sideways 0.6 (trained, reference)", g.Command(vy=0.6))
    hand = replace(g.RECOMMENDED, name="hand-tuned (green)")
    opt = sd.optimized_dodges()
    names = ["OPTIMIZED fastest (blue)", "OPTIMIZED steadiest (purple)"]
    opt = [replace(d, name=n) for d, n in zip(opt, names)]
    return [still, trained, hand, *opt, g.Dodge("sustained 1.5 (orange)", g.Command(vy=1.5, freq=4.0))]


def smooth(x: np.ndarray, n: int = 5) -> np.ndarray:
    if len(x) < n:
        return x
    k = np.ones(n) / n
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)


def measure(dodge: g.Dodge, phys: g.Physics, policy: g.WTWPolicy) -> dict:
    sim = g.Go2Sim(phys)
    m, d = sim.m, sim.d
    ctl = g.DodgeController(dodge, sim, policy)
    weight = float(sum(m.body_mass[b] for b in range(m.nbody) if b == sim.trunk or _is_descendant(m, b, sim.trunk))) * G
    acc_adr = m.sensor_adr[m.sensor("imu_acc").id]
    lo, hi = m.jnt_range[[m.joint(n).id for n in g.JOINTS]].T

    foot_f, tau_frac, sat, dq_frac, ddq, acc, margin, touch_v = [], [], [], [], [], [], [], []
    in_contact = np.ones(4, bool)
    window_start = dodge.t0 - 0.2           # measure from just before the trigger to the end
    force6 = np.zeros(6)
    vel6 = np.zeros(6)
    while sim.t < dodge.total - 1e-9:
        q_t, kp, kd, phase = ctl.targets(sim.t)
        for _ in range(sim.substeps):
            sim.apply_pd(q_t, kp, kd)
            mujoco.mj_step(m, d)
            if sim.t < window_start:
                continue
            f = np.zeros(4)
            for k in range(d.ncon):
                c = d.contact[k]
                for j, fg in enumerate(sim.feet):
                    if fg in (c.geom1, c.geom2) and sim.floor in (c.geom1, c.geom2):
                        mujoco.mj_contactForce(m, d, k, force6)
                        f[j] += force6[0]
            now = f > 1.0
            for j in np.nonzero(now & ~in_contact)[0]:      # touchdown: foot speed into the floor
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, sim.feet[j], vel6, 0)
                touch_v.append(max(0.0, -vel6[5]))
            in_contact = now
            tau = d.actuator_force[sim.act]
            foot_f.append(f / weight)
            tau_frac.append(np.abs(tau) / sim.tau_max)
            sat.append(np.abs(tau) >= 0.98 * sim.tau_max)
            dq = sim.dq
            dq_frac.append(np.abs(dq) / g.MAX_SPEED)
            ddq.append(np.abs(d.qacc[sim.vadr]))
            a = d.sensordata[acc_adr:acc_adr + 3]
            acc.append(np.linalg.norm(sim.rot() @ a - np.array([0, 0, G])) / G)
            q = sim.q
            margin.append(np.minimum(q - lo, hi - q))
        sim.t += g.POLICY_DT
        ctl.record(sim.t, phase)

    foot_f, tau_frac, dq_frac, ddq = map(np.array, (foot_f, tau_frac, dq_frac, ddq))
    acc = np.array(acc)
    margin = np.array(margin)                       # [steps, 12]
    worst_j = int(np.argmin(margin.min(0)))
    near = margin[:, worst_j] < 0.05
    tv = np.array(touch_v) if touch_v else np.zeros(1)
    return {
        "fell": bool(ctl.fell),
        "foot_force_bw": float(smooth(foot_f).max()), "foot_force_bw_raw": float(foot_f.max()),
        "touchdown_ms": float(max(touch_v) if touch_v else 0.0),
        "torque_frac": float(smooth(tau_frac).max()), "saturated_pct": float(100 * np.mean(np.array(sat))),
        "speed_frac": float(smooth(dq_frac).max()),
        "joint_acc": float(smooth(ddq).max()),
        "trunk_acc_g": float(smooth(acc[:, None])[:, 0].max()), "trunk_acc_g_raw": float(acc.max()),
        "limit_margin": float(margin.min()),
        "limit_joint": g.JOINTS[worst_j],
        "limit_ms_within_0.05rad": float(near.sum() * m.opt.timestep * 1000),
        "touchdown_median": float(np.median(tv)), "touchdown_p95": float(np.percentile(tv, 95)),
    }


def _is_descendant(m, b, root) -> bool:
    while b > 0:
        b = m.body_parentid[b]
        if b == root:
            return True
    return False


COLS = [("foot_force_bw", "foot F [BW]", "{:5.2f}"), ("touchdown_median", "tdn med", "{:5.2f}"),
        ("touchdown_p95", "tdn p95", "{:5.2f}"),
        ("torque_frac", "torque/lim", "{:5.2f}"), ("saturated_pct", "sat %", "{:5.1f}"),
        ("speed_frac", "speed/max", "{:5.2f}"), ("joint_acc", "joint acc", "{:6.0f}"),
        ("trunk_acc_g", "trunk g", "{:5.2f}"), ("limit_margin", "lim margin", "{:5.2f}")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rand", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results/load_check.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    policy = g.WTWPolicy()
    rng = np.random.default_rng(args.seed)
    physics = [g.Physics()] + [g.random_physics(rng, g.Physics()) for _ in range(args.rand)]
    results = {}
    for dg in references_and_dodges():
        results[dg.name] = [measure(dg, p, policy) for p in physics]

    def table(title, agg):
        print(f"\n{title}")
        print(f"{'':40s}" + "".join(f"{h:>12s}" for _, h, _ in COLS))
        for name, ms in results.items():
            row = agg(ms)
            print(f"{name:40s}" + "".join(f"{fmt.format(row[k]):>12s}" for k, _, fmt in COLS)
                  + ("   FELL" if row["fell"] else ""))

    table("nominal physics (friction 0.7)", lambda ms: ms[0])

    def worst(ms):
        w = {k: max(m[k] for m in ms) for k, _, _ in COLS}
        w["limit_margin"] = min(m["limit_margin"] for m in ms)
        w["fell"] = any(m["fell"] for m in ms)
        return w

    table(f"worst case over nominal + {args.rand} randomized physics", worst)
    print("\nraw (unsmoothed) peaks, nominal: foot force [BW] / trunk [g];  joint closest to its range limit (worst run)")
    for name, ms in results.items():
        w = min(ms, key=lambda r: r["limit_margin"])
        print(f"  {name:40s} {ms[0]['foot_force_bw_raw']:6.2f} / {ms[0]['trunk_acc_g_raw']:6.2f}   "
              f"{w['limit_joint']:16s} margin {w['limit_margin']:+.3f} rad, "
              f"{w['limit_ms_within_0.05rad']:.0f} ms within 0.05 rad")
    out = {name: {"nominal": ms[0], "worst": worst(ms), "runs": ms} for name, ms in results.items()}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
