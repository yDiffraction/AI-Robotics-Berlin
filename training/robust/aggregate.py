#!/usr/bin/env python3
"""Merge per-fold results of run_grid.sh: table with bootstrap CIs (resampling test clips)
and, with --groups, balanced throw/feint accuracy per recording session.

usage: python aggregate.py runs_robust [--groups]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from robust_cv import summarize  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "runs_robust")
by_group = "--groups" in sys.argv
keys = ["bal_acc", "recall_throw", "fp_feint", "fp_idle", "acc_tf", "prefix_tf@-200ms", "swap_tf", "stress_shift10"]
out = {}
print(f"{'tag':28s} " + " ".join(f"{k[:14]:>14s}" for k in keys) + "  thr/fei/idl  folds  bal_acc CI95")
for d in sorted(root.iterdir()):
    fs = sorted(d.glob("result*.json"))
    if not fs:
        continue
    rows = [r for f in fs for r in json.loads(f.read_text())["rows"]]
    s = summarize(rows)
    rng = np.random.default_rng(0)
    boot = [summarize([rows[i] for i in rng.integers(0, len(rows), len(rows))])["bal_acc"] for _ in range(300)]
    s["ci"] = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
    s["folds"] = len(fs)
    if by_group:
        s["groups"] = {g: summarize([r for r in rows if r.get("group") == g])
                       for g in sorted({r.get("group", "") for r in rows})}
    out[d.name] = s
    print(f"{d.name:28s} " + " ".join(f"{s.get(k, float('nan')):14.3f}" for k in keys) +
          f"  {s['n_throw']:3d}/{s['n_feint']:3d}/{s['n_idle']:3d}  {len(fs):5d}  [{s['ci'][0]:.2f},{s['ci'][1]:.2f}]")
    if by_group:
        for g, gs in s["groups"].items():
            print(f"    {g[:40]:40s} thr/fei/idl {gs['n_throw']:3d}/{gs['n_feint']:3d}/{gs['n_idle']:3d}"
                  f"  recall_throw {gs['recall_throw']:.2f}  fp_feint {gs['fp_feint']:.2f}  fp_idle {gs['fp_idle']:.2f}")
json.dump(out, open(root / "summary.json", "w"), indent=1, default=float)
