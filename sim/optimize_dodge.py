"""Search for the fastest sideways dodge (time to 30 cm) that still ends in a stable stand.

Stage 1  random search over all dodge knobs; every candidate runs on the same physics set
         (nominal + N randomized: friction, trunk mass, motor strength, latency). Candidates
         that are not stable in every run are discarded; the rest are ranked by worst-case
         time to 30 cm, then by peak body tilt during the motion.
Stage 2  local refinement around the best candidates.
Stage 3  validation of the finalists on fresh, harsher randomization and on perturbed copies
         of their own settings (+-10 %), so a winner on a knife edge does not survive.

  cd sim && python optimize_dodge.py            # ~3-5 min on 14 cores
  -> results/optimize.json, and the finalists printed as Dodge(...) lines

--safe: every run uses deployment-like harsh physics (sensor noise, joint-target clamp, random gait
phase at the trigger, +3 kg, weak motors, 30 ms latency) and also measures mechanical load; a run
only counts if it passes ALL safety limits of validate_dodge.py (joint margin, foot force, touchdown
speed, trunk acceleration, torque, tilt) on top of not falling and ending still.
  cd sim && python validate_dodge.py --reference     # once: sets the load limits
  cd sim && python optimize_dodge.py --safe          # -> results/optimize_safe.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import go2_dodge_sim as g

# name: (low, high)
SPACE = {
    "kick_vy": (1.0, 4.0),       # over-command at the start [m/s]
    "kick_s": (0.1, 0.5),        # its duration [s]
    "hold_vy": (0.0, 1.5),       # command after the kick until the brake [m/s]
    "dodge_s": (0.2, 0.8),       # total command time [s] (>= kick_s)
    "freq": (2.0, 4.0),          # step frequency [Hz] (trained range)
    "stance_width": (0.25, 0.45),
    "footswing": (0.04, 0.15),
    "height": (-0.10, 0.05),     # body height offset [m] (trained -0.25..0.15)
    "brake_vy": (-1.0, 0.0),     # counter-command at the start of the brake [m/s]
    "brake_s": (0.0, 0.3),
}

_POLICY = None
_PHYS: list[g.Physics] = []
_LIM: dict | None = None        # safety limits (safe mode) or None


def _init(phys, lim=None):
    global _POLICY, _PHYS, _LIM
    torch.set_num_threads(1)
    _POLICY, _PHYS, _LIM = g.WTWPolicy(), phys, lim


def to_dodge(p: dict, name: str = "candidate") -> g.Dodge:
    dodge_s = max(p["dodge_s"], p["kick_s"])
    cmd = g.Command(vy=p["hold_vy"], freq=p["freq"], stance_width=p["stance_width"],
                    footswing=p["footswing"], height=p["height"])
    return g.Dodge(name, cmd, kick_vy=p["kick_vy"], kick_s=p["kick_s"], dodge=dodge_s,
                   brake_vy=p["brake_vy"] if p["brake_s"] > 0.02 else None, brake_s=p["brake_s"])


def evaluate(p: dict) -> dict:
    d = to_dodge(p)
    safe = _LIM is not None
    ms = [g.metrics(g.run(d, ph, _POLICY, loads=safe)) for ph in _PHYS]
    t30 = [m["t_30cm"] for m in ms]
    if safe:
        import validate_dodge as v
        viol = [v.violations(m, _LIM) for m in ms]
        ok = sum(not x for x in viol)
        extra = {"margin_worst": float(min(m["limit_margin"] for m in ms)),
                 "foot_worst": float(max(m["foot_force_bw"] for m in ms)),
                 "touch_worst": float(max(m["touchdown_p95"] for m in ms)),
                 "acc_worst": float(max(m["trunk_acc_g"] for m in ms)),
                 "why": sorted({w.split(" ")[0] for x in viol for w in x})}
    else:
        ok = sum(m["stable_end"] for m in ms)
        extra = {}
    return {**extra,
        "params": p, "ok": ok, "n": len(ms),
        "t30_worst": float(max(9.0 if math.isnan(t) else t for t in t30)),
        "t30_nominal": float(9.0 if math.isnan(t30[0]) else t30[0]),
        "tilt_worst": float(max(m["peak_tilt_deg"] for m in ms)),
        "yaw_worst": float(max(abs(m["yaw_drift_deg"]) for m in ms)),
        "y_final_med": float(np.median([m["y_final"] for m in ms])),
    }


def score(r: dict) -> tuple:
    """Lower is better: stable everywhere first, then worst-case speed, then tilt."""
    return (r["n"] - r["ok"], round(r["t30_worst"], 2), r["tilt_worst"])


def sample(rng) -> dict:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SPACE.items()}


def perturb(p: dict, rng, scale: float) -> dict:
    q = {}
    for k, (lo, hi) in SPACE.items():
        q[k] = float(np.clip(p[k] + rng.normal(0, scale * (hi - lo)), lo, hi))
    return q


def physics_set(rng, n: int, harsh: bool = False, safe: bool = False) -> list[g.Physics]:
    base = g.Physics()
    if safe:
        lead = [] if harsh else [g.Physics(obs_noise=1.0, target_margin=0.1)]
        return lead + [g.harsh_physics(rng) for _ in range(n)]
    if not harsh:
        return [base] + [g.random_physics(rng, base) for _ in range(n)]
    return [replace(base, friction=rng.uniform(0.35, 1.1), mass_add=rng.uniform(-1.5, 3.0),
                    motor_strength=rng.uniform(0.8, 1.0), latency=rng.uniform(0.0, 0.03)) for _ in range(n)]


def table(rows: list[dict], k: int) -> None:
    print(f"  {'ok':>5s} {'t30 worst':>9s} {'t30 nom':>7s} {'tilt°':>5s} {'yaw°':>5s} {'y end':>5s}  params")
    for r in rows[:k]:
        p = r["params"]
        print(f"  {r['ok']:>2d}/{r['n']:<2d} {r['t30_worst']:9.2f} {r['t30_nominal']:7.2f} {r['tilt_worst']:5.1f} "
              f"{r['yaw_worst']:5.0f} {r['y_final_med']:5.2f}  "
              f"kick {p['kick_vy']:.2f}/{p['kick_s']:.2f}s hold {p['hold_vy']:.2f} to {max(p['dodge_s'], p['kick_s']):.2f}s "
              f"f{p['freq']:.1f} w{p['stance_width']:.2f} sw{p['footswing']:.2f} h{p['height']:+.2f} "
              f"brake {p['brake_vy']:.2f}/{p['brake_s']:.2f}s"
              + (f"  | margin {r['margin_worst']:.3f} foot {r['foot_worst']:.2f} touch {r['touch_worst']:.2f} "
                 f"acc {r['acc_worst']:.2f} {','.join(r['why'])}" if "margin_worst" in r else ""))


def run_pool(phys, params, workers, lim=None):
    with Pool(workers, initializer=_init, initargs=(phys, lim)) as pool:
        return pool.map(evaluate, params, chunksize=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--rand", type=int, default=12)
    ap.add_argument("--refine", type=int, default=12, help="candidates to refine")
    ap.add_argument("--refine-n", type=int, default=30, help="perturbations per refined candidate")
    ap.add_argument("--final", type=int, default=8)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--safe", action="store_true", help="harsh physics + hard load limits (see module doc)")
    ap.add_argument("--val-n", type=int, default=None, help="validation runs per finalist (default 40, safe 60)")
    args = ap.parse_args()
    args.out = args.out or ("results/optimize_safe.json" if args.safe else "results/optimize.json")
    lim = None
    if args.safe:
        import validate_dodge as v
        lim = v.limits()
        print(f"safe mode, limits: {lim}")
    val_n = args.val_n or (60 if args.safe else 40)
    rng = np.random.default_rng(args.seed)
    search_phys = physics_set(rng, args.rand, safe=args.safe)

    # anchors: the earlier hand-tuned dodge and the trained-limit baseline
    anchors = [dict(kick_vy=2.0, kick_s=0.3, hold_vy=1.0, dodge_s=0.5, freq=3.0, stance_width=0.33,
                    footswing=0.08, height=0.0, brake_vy=0.0, brake_s=0.0),
               dict(kick_vy=0.6, kick_s=0.1, hold_vy=0.6, dodge_s=0.8, freq=3.0, stance_width=0.33,
                    footswing=0.08, height=0.0, brake_vy=0.0, brake_s=0.0)]

    print(f"stage 1: {args.samples} random candidates x {len(search_phys)} physics")
    rows = run_pool(search_phys, anchors + [sample(rng) for _ in range(args.samples)], args.workers, lim)
    rows.sort(key=score)
    table(rows, 10)

    print(f"\nstage 2: refine top {args.refine} x {args.refine_n}")
    seeds = [r["params"] for r in rows[:args.refine]]
    cand = [perturb(p, rng, s) for p in seeds for s in np.linspace(0.02, 0.12, args.refine_n)]
    rows2 = run_pool(search_phys, cand, args.workers, lim)
    allrows = sorted(rows + rows2, key=score)
    table(allrows, 10)

    print(f"\nstage 3: validate top {args.final}: {val_n} fresh harsh physics + 20 perturbed (+-10%) settings each")
    val_phys = physics_set(np.random.default_rng(args.seed + 100), val_n, harsh=True, safe=args.safe)
    finalists = []
    seen = set()
    for r in allrows:
        key = tuple(round(v, 3) for v in r["params"].values())
        if key not in seen and r["ok"] == r["n"]:
            seen.add(key)
            finalists.append(r)
        if len(finalists) == args.final:
            break
    val = run_pool(val_phys, [f["params"] for f in finalists], args.workers, lim)
    prng = np.random.default_rng(args.seed + 200)
    neigh_params = [perturb(f["params"], prng, 0.05) for f in finalists for _ in range(20)]
    neigh = run_pool(search_phys, neigh_params, args.workers, lim)
    report = []
    for i, (f, v) in enumerate(zip(finalists, val)):
        nb = neigh[i * 20:(i + 1) * 20]
        report.append({"search": f, "validation": v,
                       "neighbours_ok": sum(n["ok"] == n["n"] for n in nb), "neighbours_n": len(nb),
                       "neighbours_t30_worst": max(n["t30_worst"] for n in nb)})
    report.sort(key=lambda r: (r["validation"]["n"] - r["validation"]["ok"], r["neighbours_n"] - r["neighbours_ok"],
                               r["validation"]["t30_worst"], r["validation"]["tilt_worst"]))
    print(f"  {'val ok':>6s} {'nbr ok':>6s} {'t30 nom':>7s} {'t30 worst(val)':>14s} {'tilt°(val)':>10s} {'yaw°':>5s}  params")
    for r in report:
        v, p = r["validation"], r["search"]["params"]
        print(f"  {v['ok']:>2d}/{v['n']:<3d} {r['neighbours_ok']:>2d}/{r['neighbours_n']:<3d} {r['search']['t30_nominal']:7.2f} "
              f"{v['t30_worst']:14.2f} {v['tilt_worst']:10.1f} {v['yaw_worst']:5.0f}  "
              + json.dumps({k: round(x, 2) for k, x in p.items()}))
    best = report[0]["search"]["params"]
    print("\nbest as code:\n  " + repr(to_dodge(best, "OPTIMIZED")).replace("Command(", "\n    Command("))

    Path(args.out).write_text(json.dumps({"space": SPACE, "stage1_2_top": allrows[:50], "final": report},
                                         indent=1, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
