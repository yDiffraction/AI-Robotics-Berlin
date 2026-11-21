"""How much lateral room the SAFE dodge needs: its largest sideways travel (braking included) over
harsh, deployment-like runs. The side check requires this plus a margin before it lets the dog go.

  cd sim/avoid && python dodge_travel.py --runs 300     -> ../results/dodge_travel.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import go2_dodge_sim as g  # noqa: E402

_POL = None


def _init():
    global _POL
    torch.set_num_threads(1)
    _POL = g.WTWPolicy()


def _one(args):
    dodge, phys = args
    res = g.run(dodge, phys, _POL)
    L = res["log"]
    after = L["t"] >= 0
    return {"max_y": float(L["y"][after].max()), "final_y": float(L["y"][-1]),
            "max_abs_x": float(np.abs(L["x"][after]).max()), "fell": res["fell"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    import show_dodges as sd
    safe = sd.safe_dodge()
    rng = np.random.default_rng(args.seed)
    jobs = [(safe, g.harsh_physics(rng)) for _ in range(args.runs)]
    with Pool(14, initializer=_init) as pool:
        rows = pool.map(_one, jobs, chunksize=4)
    ys = np.array([r["max_y"] for r in rows])
    out = {"runs": args.runs, "max_y_p50": float(np.median(ys)), "max_y_p99": float(np.percentile(ys, 99)),
           "max_y_max": float(ys.max()), "max_abs_x_max": float(max(r["max_abs_x"] for r in rows)),
           "falls": sum(r["fell"] for r in rows)}
    print(json.dumps(out, indent=1))
    (HERE.parent / "results/dodge_travel.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
