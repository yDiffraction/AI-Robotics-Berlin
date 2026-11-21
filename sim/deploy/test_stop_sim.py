"""What happens to the robot when the supervisor stops it at a random moment of the SAFE dodge?
Pure MuJoCo (no DDS), harsh randomized physics as in validate_dodge.py.

At a random time from 0.3 s before the trigger to the end of the stand-up, the controller stops.
For a random detection delay (0-80 ms: e-stop ~10 ms, crash/hang up to 70 ms) the robot keeps the
last command, then one of these stop modes applies until 2.5 s later:
  damp      kp=0, kd=5   (what the supervisor sends; Unitree's damping mode)
  damp_kd2 / damp_kd8    kd sensitivity
  passive   zero torque   (power cut, or the firmware going limp without commands)
  hold      last command forever (the firmware keeps the last target and nobody sends anything)

  python test_stop_sim.py --n 100            # ~5-10 min
Writes ../results/stop_sim.json; prints per-mode peaks compared with the dodge's own load references.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import go2_dodge_sim as g  # noqa: E402
import show_dodges as sd  # noqa: E402
import validate_dodge as vd  # noqa: E402

MODES = {"damp": (0.0, 5.0), "damp_kd2": (0.0, 2.0), "damp_kd8": (0.0, 8.0), "passive": (0.0, 0.0), "hold": None}
AFTER = 2.5


def run_stop(dodge: g.Dodge, phys: g.Physics, policy: g.WTWPolicy, t_stop: float, detect: float, mode: str) -> dict:
    if phys.pre_offset:
        dodge = replace(dodge, pre=dodge.pre + phys.pre_offset)
    sim = g.Go2Sim(phys)
    ctl = g.DodgeController(dodge, sim, policy)
    t_stop += dodge.t0                     # t_stop is relative to the trigger
    last = (g.DEFAULT_Q, dodge.stand_kp, dodge.stand_kd)
    meter = None
    vz_min, tilt_max, trunk_hit_v, stop_speed = 0.0, 0.0, 0.0, 0.0
    trunk_was_on_floor = False
    phase_at_stop = None
    while sim.t < t_stop + AFTER - 1e-9:
        if sim.t < t_stop:
            qt, kp, kd, ph = ctl.targets(sim.t)
            last = (qt, kp, kd)
            phase_at_stop = ph
        elif sim.t < t_stop + detect or MODES[mode] is None:
            qt, kp, kd = last
        else:
            kp, kd = MODES[mode]
            qt = sim.q
        if meter is None and sim.t >= t_stop:
            meter = g.LoadMeter(sim)
        for _ in range(sim.substeps):
            sim.apply_pd(qt, kp, kd)
            mujoco.mj_step(sim.m, sim.d)
            if meter is not None:
                meter.sample()
                vz = sim.d.qvel[sim.bv + 2]
                vz_min = min(vz_min, vz)
                on = sim.trunk_on_floor()
                if on and not trunk_was_on_floor:
                    trunk_hit_v = max(trunk_hit_v, -vz)
                trunk_was_on_floor = on
                r, p, _ = sim.rpy()
                tilt_max = max(tilt_max, abs(r), abs(p))
                q, dq = sim.q, sim.dq          # joint speed while at a mechanical stop (folding legs)
                at_stop = np.minimum(q - sim.q_lo, sim.q_hi - q) < 0.02
                if at_stop.any():
                    stop_speed = max(stop_speed, float(np.abs(dq[at_stop]).max()))
        sim.t += g.POLICY_DT
    r, p, _ = sim.rpy()
    return {"mode": mode, "t_stop": t_stop - dodge.t0, "detect": detect, "phase": phase_at_stop,
            **meter.summary(), "trunk_down_speed": -vz_min, "trunk_hit_speed": trunk_hit_v,
            "tilt_max_deg": math.degrees(tilt_max), "speed_at_joint_stop": stop_speed, "on_side_end": bool(max(abs(r), abs(p)) > 1.0),
            "z_end": float(sim.pos[2])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--modes", default=",".join(MODES))
    a = ap.parse_args()
    torch.set_num_threads(1)
    dodge = sd.safe_dodge()
    assert dodge is not None, "no SAFE dodge (results/safe_dodge.json)"
    policy = g.WTWPolicy()
    rng = np.random.default_rng(a.seed)
    ref = json.loads(vd.REF_PATH.read_text())
    end_rel = dodge.dodge + dodge.brake + dodge.stand_ramp + 0.3
    rows = []
    for i in range(a.n):
        phys = g.harsh_physics(rng)
        t_stop = float(rng.uniform(-0.3, end_rel))
        detect = float(rng.uniform(0.0, 0.08))
        for mode in a.modes.split(","):
            r = run_stop(dodge, phys, policy, t_stop, detect, mode)
            r["run"] = i
            rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{a.n}", flush=True)
    print(f"\n{'mode':10s} {'on side':>8s} {'trunk g p95/max':>16s} {'foot BW max':>11s} {'trunk hits floor':>17s} "
          f"{'hit speed max':>13s} {'speed at joint stop':>19s} {'z end':>6s}")
    summary = {}
    for mode in a.modes.split(","):
        rs = [r for r in rows if r["mode"] == mode]
        acc = np.array([r["trunk_acc_g"] for r in rs])
        hit = np.array([r["trunk_hit_speed"] for r in rs])
        s = {"n": len(rs), "on_side": sum(r["on_side_end"] for r in rs),
             "trunk_acc_p95": float(np.percentile(acc, 95)), "trunk_acc_max": float(acc.max()),
             "foot_force_max": max(r["foot_force_bw"] for r in rs),
             "trunk_hits": int((hit > 0).sum()), "hit_speed_max": float(hit.max()),
             "speed_at_joint_stop_max": max(r["speed_at_joint_stop"] for r in rs),
             "z_end_median": float(np.median([r["z_end"] for r in rs]))}
        summary[mode] = s
        print(f"{mode:10s} {s['on_side']:>4d}/{s['n']:<3d} {s['trunk_acc_p95']:7.2f}/{s['trunk_acc_max']:<7.2f} "
              f"{s['foot_force_max']:11.2f} {s['trunk_hits']:>9d}/{s['n']:<7d} {s['hit_speed_max']:10.2f} m/s "
              f"{s['speed_at_joint_stop_max']:13.1f} rad/s {s['z_end_median']:6.2f}")
    lim = ref["limits"]
    print(f"\nreference = 1.25 x worst normal sideways walking under harsh physics: {lim}")
    out = HERE.parent / "results" / "stop_sim.json"
    out.write_text(json.dumps({"summary": summary, "reference_limits": lim, "rows": rows}, indent=1, default=float))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
