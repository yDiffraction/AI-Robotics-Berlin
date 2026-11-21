#!/usr/bin/env python3
"""Summarise finished runs (~/throwdet/runs/*/result.json) into one comparison table.

For each run (out-of-fold predictions on our own camera, never used for training):
  AUC, balanced accuracy at threshold 0, and the live operating point: the threshold with the
  highest motion recall whose false-alarm rate is <= --max-fa alarms per minute on the
  continuous idle-recording stream and <= --max-fpr on the idle clips (pauses + idle recording).
Also per-recording recall (which recordings are hard) and DailyDVS held-out AUC.

  python throwdet/analyze.py                  # all runs
  python throwdet/analyze.py e01_all_v1 --detail
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def rec_of(fname):
    m = re.search(r"-d(\d\d_\d{6})-", fname)
    return m.group(1) if m else "?"


def summarise(path, max_fa, max_fpr):
    r = json.load(open(path))
    res = r.get("own_res")
    if not res:
        return None
    y = np.array([x["y"] for x in res]); s = np.array([x["score"] for x in res])
    pos, neg = s[y == 1], s[y == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    grid = {float(k): v for k, v in r.get("own_thr_grid", {}).items()}
    # live operating point from the stored grid (stream FA) refined with clip-level rates
    best = None
    for thr in sorted(grid):
        g = grid[thr]
        fa = g.get("stream_fa_per_min")
        fpr = float((neg > thr).mean())
        if fa is not None and fa <= max_fa and fpr <= max_fpr:
            tpr = float((pos > thr).mean())
            if best is None or tpr > best["tpr"]:
                best = {"thr": thr, "tpr": tpr, "fpr": fpr, "fa_per_min": fa}
    dd = r["folds"].get("fold0", r["folds"].get("final", {})).get("dd_test", {})
    out = {"tag": Path(path).parent.name, "n": len(y), "auc": auc,
           "bal_acc@0": float(((pos > 0).mean() + (neg <= 0).mean()) / 2),
           "live": best, "dd_auc": dd.get("auc"), "grid": grid}
    by = defaultdict(list)
    for x in res:
        by[(rec_of(x["fname"]), x["kind"])].append(x["score"])
    out["by_rec"] = {f"{k[0]}/{k[1]}": (len(v), [round(float(np.mean(np.array(v) > t)), 2) for t in (0.0, 0.2, 0.4)])
                     for k, v in sorted(by.items())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--runs", default=str(Path.home() / "throwdet/runs"))
    ap.add_argument("--max-fa", type=float, default=1.0, help="max false alarms per minute on the idle stream")
    ap.add_argument("--max-fpr", type=float, default=0.15, help="max fraction of idle clips with an alarm")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()
    paths = sorted(Path(args.runs).glob("*/result.json"))
    if args.tags:
        paths = [p for p in paths if p.parent.name in args.tags]
    rows = [s for s in (summarise(p, args.max_fa, args.max_fpr) for p in paths) if s]
    rows.sort(key=lambda s: -(s["live"]["tpr"] if s["live"] else 0))
    print(f"| run | AUC own | bal.acc @0 | live thr | recall @live | idle-clip FPR @live | FA/min stream @live | DailyDVS AUC |")
    print("|---|---|---|---|---|---|---|---|")
    for s in rows:
        lv = s["live"] or {}
        f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
        print(f"| {s['tag']} | {f(s['auc'])} | {f(s['bal_acc@0'])} | {f(lv.get('thr'), 1)} | {f(lv.get('tpr'))} | "
              f"{f(lv.get('fpr'))} | {f(lv.get('fa_per_min'), 2)} | {f(s['dd_auc'])} |")
    if args.detail:
        for s in rows:
            print(f"\n## {s['tag']}\nthreshold grid: {json.dumps(s['grid'])}")
            print("detection rate per recording/kind at thr 0 / 0.2 / 0.4:")
            for k, (n, v) in s["by_rec"].items():
                print(f"  {k:24s} n={n:4d}  {v}")


if __name__ == "__main__":
    main()
