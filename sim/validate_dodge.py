"""Validate dodges on many harsh, deployment-like runs and report failure rate and mechanical load.

Harsh runs (go2_dodge_sim.harsh_physics): friction 0.35-1.1, trunk mass -1..+3 kg, motors 80-100 %,
latency 0-30 ms, sensor noise (legged_gym levels), joint targets clamped 0.1 rad inside the joint
range (as the deploy script will do), and the trigger at a random gait phase.

A run FAILS if the dog falls, does not end standing still on four feet, or breaks a safety limit:
  limit_margin >= 0.05 rad        no joint closer than ~3 deg to its mechanical stop
  foot_force_bw <= LIMIT          peak foot force (10 ms avg), body weights
  touchdown_p95 <= LIMIT          foot speed into the ground at touchdown [m/s]
  trunk_acc_g   <= LIMIT          trunk acceleration without gravity (10 ms avg) [g]
  torque_frac   <= 0.97           no motor at its torque limit
  peak_tilt_deg <= 20             body roll/pitch during the motion
The force/touchdown/acceleration limits are set relative to walking sideways at the trained 0.6 m/s
under the same harsh conditions (see results/safety_reference.json).

  cd sim && python validate_dodge.py --runs 500             # the SAFE dodge + references
  cd sim && python validate_dodge.py --reference --runs 60  # (re)measure the reference + set the limits
"""
from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import go2_dodge_sim as g

HERE = Path(__file__).resolve().parent
REF_PATH = HERE / "results/safety_reference.json"
LOAD_KEYS = ("foot_force_bw", "touchdown_p95", "trunk_acc_g", "torque_frac", "speed_frac", "limit_margin", "peak_tilt_deg")
FIXED_LIMITS = {"limit_margin_min": 0.05, "torque_frac_max": 0.97, "peak_tilt_deg_max": 20.0}
HEADROOM = 1.25    # dodge may load feet/trunk up to 1.25x the worst of normal sideways walking

_POL = None


def limits() -> dict:
    lim = dict(FIXED_LIMITS)
    if REF_PATH.exists():
        lim.update(json.loads(REF_PATH.read_text())["limits"])
    return lim


def violations(m: dict, lim: dict) -> list[str]:
    v = []
    if m["fell"]:
        v.append("fell")
    elif not m["stable_end"]:
        v.append("unsettled end")
    if m["limit_margin"] < lim["limit_margin_min"]:
        v.append(f"joint margin {m['limit_margin']:.3f}")
    for k in ("foot_force_bw", "touchdown_p95", "trunk_acc_g", "torque_frac", "peak_tilt_deg"):
        if f"{k}_max" in lim and m[k] > lim[f"{k}_max"]:
            v.append(f"{k} {m[k]:.2f}")
    if math.isnan(m["t_30cm"]) and "fell" not in v:
        v.append("never reached 30 cm")
    return v


def _init():
    global _POL
    torch.set_num_threads(1)
    _POL = g.WTWPolicy()


def _one(args):
    dodge, phys = args
    m = g.metrics(g.run(dodge, phys, _POL, loads=True))
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()}


def run_many(dodge: g.Dodge, n: int, seed: int, workers: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    jobs = [(dodge, g.harsh_physics(rng)) for _ in range(n)]
    with Pool(workers, initializer=_init) as pool:
        return pool.map(_one, jobs, chunksize=4)


def report(name: str, ms: list[dict], lim: dict) -> dict:
    fails = [(i, violations(m, lim)) for i, m in enumerate(ms)]
    bad = [(i, v) for i, v in fails if v]
    n = len(ms)
    upper = 3.0 / n if not bad else None          # rule of three: 95 % upper bound with 0 failures
    t30 = np.array([m["t_30cm"] for m in ms], float)
    summ = {
        "runs": n, "failures": len(bad),
        "failure_rate_95_upper": upper,
        "t30_median": float(np.nanmedian(t30)), "t30_p95": float(np.nanpercentile(t30, 95)),
        "t30_worst": float(np.nanmax(t30)) if not np.all(np.isnan(t30)) else float("nan"),
        **{f"{k}_worst": float(min(m[k] for m in ms) if k == "limit_margin" else max(m[k] for m in ms))
           for k in LOAD_KEYS},
        "failure_reasons": {str(i): v for i, v in bad[:20]},
    }
    rate = f"0/{n} failed (95% upper bound {100 * upper:.1f} %)" if not bad else f"{len(bad)}/{n} FAILED"
    print(f"\n{name}: {rate}")
    print(f"  30 cm: median {summ['t30_median']:.2f} s, p95 {summ['t30_p95']:.2f} s, worst {summ['t30_worst']:.2f} s")
    print("  worst load: " + ", ".join(f"{k} {summ[k + '_worst']:.2f}" for k in LOAD_KEYS))
    for i, v in bad[:5]:
        print(f"  run {i}: {', '.join(v)}")
    return summ


SAFE_PATH = HERE / "results/safe_dodge.json"
SCALES = (1.0, 0.9, 0.8, 0.7, 0.6)


def scaled(p: dict, s: float) -> dict:
    """Same dodge with all sideways speed commands (kick, hold, brake) scaled by s."""
    return {**p, "kick_vy": p["kick_vy"] * s, "hold_vy": p["hold_vy"] * s, "brake_vy": p["brake_vy"] * s}


def scale_sweep(args) -> None:
    """Safety knob: scale the two best safe finalists down until a version passes every harsh run."""
    import optimize_dodge as o
    final = json.loads((HERE / "results/optimize_safe.json").read_text())["final"]
    bases = sorted(final, key=lambda f: (f["validation"]["n"] - f["validation"]["ok"],
                                         f["neighbours_n"] - f["neighbours_ok"]))[:2]
    lim = limits()
    print(f"limits: {lim}\n{args.runs} harsh runs per version (seed {args.seed})")
    rows = []
    for bi, b in enumerate(bases):
        for s in SCALES:
            p = scaled(b["search"]["params"], s)
            name = f"base {bi + 1} x{s:.1f}"
            summ = report(name, run_many(o.to_dodge(p, name), args.runs, args.seed, args.workers), lim)
            rows.append({"base": bi + 1, "scale": s, "params": p, **summ})
            if summ["failures"] == 0:
                break                   # slower versions of this base are not needed
    clean = [r for r in rows if r["failures"] == 0]
    if not clean:
        print("\nno version passed every run; extend SCALES")
        return
    best = min(clean, key=lambda r: (r["t30_p95"], r["peak_tilt_deg_worst"]))
    print(f"\nSAFE = base {best['base']} x{best['scale']:.1f}: 0/{best['runs']} failures, "
          f"30 cm median {best['t30_median']:.2f} s, p95 {best['t30_p95']:.2f} s")
    SAFE_PATH.write_text(json.dumps({"chosen": best, "sweep": rows, "limits": lim}, indent=1))
    print(f"-> {SAFE_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--reference", action="store_true", help="measure normal walking and write the load limits")
    ap.add_argument("--dodges", default="safe,purple", help="comma list: safe, purple, blue, green")
    ap.add_argument("--out", default=str(HERE / "results/validate.json"))
    ap.add_argument("--scale-sweep", action="store_true",
                    help="scale the speed commands of the best safe finalists down until a version has 0 failures; "
                         "writes results/safe_dodge.json")
    args = ap.parse_args()

    if args.reference:
        refs = {"trot in place": g.Dodge("trot in place", g.Command(vy=0.0)),
                "walk sideways 0.6 (trained)": g.Dodge("walk 0.6", g.Command(vy=0.6))}
        out = {}
        for name, d in refs.items():
            ms = run_many(d, args.runs, args.seed, args.workers)
            out[name] = report(name, ms, dict(FIXED_LIMITS))
        w = out["walk sideways 0.6 (trained)"]
        lim = {f"{k}_max": round(w[f"{k}_worst"] * HEADROOM, 3) for k in ("foot_force_bw", "touchdown_p95", "trunk_acc_g")}
        REF_PATH.write_text(json.dumps({"headroom": HEADROOM, "limits": lim, "reference": out}, indent=1))
        print(f"\nlimits ({HEADROOM}x worst normal sideways walking): {lim}  -> {REF_PATH}")
        return

    if args.scale_sweep:
        scale_sweep(args)
        return

    import show_dodges as sd
    pick = {"safe": sd.safe_dodge()}
    opt = sd.optimized_dodges()
    if len(opt) >= 2:
        pick["blue"], pick["purple"] = opt[0], opt[1]
    pick["green"] = g.RECOMMENDED
    lim = limits()
    print(f"limits: {lim}")
    out = {"limits": lim}
    for key in args.dodges.split(","):
        d = pick.get(key)
        if d is None:
            print(f"(no '{key}' dodge available)")
            continue
        ms = run_many(d, args.runs, args.seed, args.workers)
        out[key] = report(f"{key}: {d.name}", ms, lim)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
