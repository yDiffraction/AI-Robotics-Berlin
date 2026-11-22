"""Screen hand-picked STILL dodge candidates (and scaled versions) on N fresh harsh runs per side.

The search in optimize_still.py ranks on few runs, so its fastest finalists overfit (1-12 of 80
fresh harsh runs failed). This screens robust candidates on many more runs before the final
500-per-side validation (optimize_still.py --validate --params results/still_screen.json).

  cd sim && python screen_still.py --runs 100      # -> results/still_screen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import go2_dodge_sim as g
import optimize_still as o
import still_dodge as s
import validate_dodge as v

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/still_screen.json"


def candidates() -> dict[str, dict]:
    fin = json.loads(o.OUT.read_text())["final"]
    safe = o.safe_anchor()
    base = {
        "F1 (search best)": fin[0]["search"]["params"],
        "F2": fin[1]["search"]["params"],
        "SAFE params, phase 0.5": {**safe, "phase0": 0.5},
        "SAFE params, phase 0.0": {**safe, "phase0": 0.0},
        "low-load (search #10)": json.loads(o.OUT.read_text())["top"][9]["params"],
    }
    out = {}
    for name, p in base.items():
        for sc in ((1.0, 0.9) if name.startswith("F") else (1.0,)):
            q = s.StillDodge.from_params(p).scaled(sc).params()
            out[f"{name} x{sc:g}"] = {k: float(q[k]) for k in o.SPACE}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    lim = v.limits()
    rng = np.random.default_rng(a.seed)
    phys = [g.harsh_physics(rng) for _ in range(a.runs)]
    rows = {}
    for name, p in candidates().items():
        r = o._split(phys, lim, p, a.workers)
        L, R = r["left"], r["right"]
        rows[name] = {k: r[k] for k in ("params", "fails", "n", "t30_p90", "load_cost")}
        rows[name].update({side: {k: x for k, x in r[side].items() if k != "runs"} for side in ("left", "right")})
        print(f"{name:28s} fails {r['fails']:>3d}/{r['n']}  t30 med {L['t30_median']:.2f}/{R['t30_median']:.2f} "
              f"p90 {L['t30_p90']:.2f}/{R['t30_p90']:.2f} worst {L['t30_worst']:.2f}/{R['t30_worst']:.2f}  "
              f"load {r['load_cost']:.2f}  tilt {max(L['tilt_worst'], R['tilt_worst']):.0f}  "
              f"y_end {min(L['y_final_min'], R['y_final_min']):.2f}-{max(L['y_final_max'], R['y_final_max']):.2f}  "
              f"x {max(L['x_worst'], R['x_worst']):.2f} yaw {max(L['yaw_worst'], R['yaw_worst']):.0f}  "
              f"{sorted(set(L['why'] + R['why']))}", flush=True)
    OUT.write_text(json.dumps({"runs_per_side": a.runs, "seed": a.seed, "rows": rows}, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
