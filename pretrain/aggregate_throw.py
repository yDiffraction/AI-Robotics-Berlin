"""Eval 5: functional tests from the runs_throw results.

  CV (grouped by recording, 5 folds): real-only vs real + N synthetic -> pooled accuracy on held-out real windows
  TSTR: synthetic only (N = 100 / 300 / 1050, pretrained; 1050 from scratch) -> accuracy on all 38 real windows
  TRTS: real-only CV models evaluated on the 150 synthetic test clips
Bootstrap CIs resample source recordings (windows of one recording stay together).

usage: python aggregate_throw.py ../runs_throw --out ../runs_throw/summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

K = 3


def windows_of(run):
    out = []
    for w in run["real_test_windows"]:
        c = np.array(w["counts"])
        out.append({"parent": w["parent"], "y": w["y"], "pred": int(c.argmax()) if c.sum() > 0 else -1})
    return out


def stats(ws, rng, n_boot=2000):
    if not ws:
        return None
    y = np.array([w["y"] for w in ws])
    p = np.array([w["pred"] for w in ws])
    par = np.array([w["parent"] for w in ws])
    hit = (y == p).astype(float)
    tf = y < 2
    up = np.unique(par)
    groups = [np.flatnonzero(par == u) for u in up]
    boot, boot_tf = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boot.append(hit[idx].mean())
        m = idx[tf[idx]]
        if len(m):
            boot_tf.append(hit[m].mean())
    cm = np.zeros((K, K), int)
    for a, b in zip(y, p):
        if b >= 0:
            cm[a, b] += 1
    return {"n": int(len(y)), "acc": float(hit.mean()), "acc_ci95": np.percentile(boot, [2.5, 97.5]).round(3).tolist(),
            "acc_throw_feint": float(hit[tf].mean()) if tf.any() else None,
            "acc_throw_feint_ci95": np.percentile(boot_tf, [2.5, 97.5]).round(3).tolist() if boot_tf else None,
            "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    runs = {p.parent.name: json.loads(p.read_text()) for p in Path(args.runs).glob("*/result.json")}
    res = {"cv": {}, "tstr": {}, "trts": {}, "synth_test": {}}
    cv = defaultdict(list)
    for tag, r in runs.items():
        if tag.startswith("cv"):
            cv[tag.split("_", 1)[1]].append((tag, r))
    for cfg, items in sorted(cv.items()):
        ws = [w for _, r in items for w in windows_of(r)]
        res["cv"][cfg] = {"folds": sorted(t for t, _ in items), **stats(ws, rng)}
        res["synth_test"][f"cv_{cfg}"] = float(np.mean([r["synth_test"]["acc_end"] for _, r in items]))
    for tag, r in sorted(runs.items()):
        if tag.startswith("tstr"):
            res["tstr"][tag] = stats(windows_of(r), rng)
            res["synth_test"][tag] = r["synth_test"]["acc_end"]
    if "s0" in cv:
        res["trts"] = {"real_only_cv_models_on_synth_test": res["synth_test"]["cv_s0"]}
    Path(args.out).write_text(json.dumps(res, indent=1))
    fmt = lambda s: (f"{s['acc']:.1%} [{s['acc_ci95'][0]:.0%}-{s['acc_ci95'][1]:.0%}]  throw/feint "
                     f"{s['acc_throw_feint']:.1%}  n={s['n']}  cm={s['confusion']}")
    print("CV on held-out real recordings (pooled over folds):")
    for cfg, s in res["cv"].items():
        print(f"  real + {cfg:6s} {fmt(s)}  folds={len(s['folds'])}")
    print("TSTR (synthetic only -> all real):")
    for tag, s in res["tstr"].items():
        print(f"  {tag:20s} {fmt(s)}")
    print("accuracy on the 150 synthetic test clips:", {k: round(v, 3) for k, v in res["synth_test"].items()})


if __name__ == "__main__":
    main()
