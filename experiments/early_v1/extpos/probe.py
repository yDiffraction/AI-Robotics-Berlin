#!/usr/bin/env python3
"""Shortcut probe: can a simple classifier tell sources / labels apart from things that are
not the throw motion? Grouped CV (by person / video / recording), balanced accuracy.

Every clip is converted to the board grid and described by
  sensor  noise floor, isolated cells, ON share, edge run length, temporal autocorrelation,
          empty frames                                   (should carry no label / source)
  level   density, peakiness, extent, centre, touched area  (motion-related, partly fair)
  map     8x8 map of where events occur over the whole clip (catches frame borders, FOV
          rectangles, fixed positions)

Questions (0.5 = cannot tell, which is the goal for the first three):
  ext_pos vs own_pos        do the new positives look like our positives?
  ext_pos vs ext_neg        within an external source, do *sensor* features give the label away?
  ext vs own                can the source be recognised at all (all clips)?

  python probe.py --own <dataset dir> --ext data/extpos/clips_dd [--ext ...] [--out probe.json]
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import cellstats as cs

SENSOR = ["cells_quiet", "isolated_frac", "on_frac", "run_len", "ac1", "ac3", "empty_frac"]
LEVEL = ["cells_mean", "cells_p90", "peak_ratio", "ext_h", "ext_w", "cx", "cy", "touched"]


def feats(path, rel):
    g = cs.to_cells(cs.read_bin(path))
    s = cs.clip_stats(g, rel)
    occ = g.any(1).sum(0).astype(np.float64)
    m8 = occ.reshape(8, 10, 8, 10).sum((1, 3)).ravel()
    m8 = m8 / max(m8.sum(), 1)
    return s, m8


def load_own(root):
    out = []
    for r in csv.DictReader(open(os.path.join(root, "labels.csv"))):
        rel = int(r["release_us"]) // cs.FRAME_US if r["label"] != "idle" else None
        s, m = feats(os.path.join(root, r["folder"], r["file"]), rel)
        out.append(dict(dom="own", y=int(r["label"] != "idle"), group="own:" + r["source"], s=s, m=m))
    return out


def load_ext(root):
    out = []
    for r in csv.DictReader(open(Path(root) / "labels.csv")):
        if r["role"] == "amb":
            continue
        rel = int(r["release_us"]) // cs.FRAME_US if r["release_us"] else None
        s, m = feats(Path(root) / r["folder"] / r["file"], rel)
        out.append(dict(dom=Path(root).name, y=int(r["role"] == "pos"), group=r["group"], s=s, m=m))
    return out


def matrix(rows, kind):
    cols = {"sensor": SENSOR, "level": LEVEL, "all": SENSOR + LEVEL}.get(kind)
    if kind == "map":
        return np.array([r["m"] for r in rows])
    return np.array([[r["s"].get(c, 0.0) for c in cols] for r in rows], np.float64)


def logreg(X, y, l2=1.0, iters=300):
    w, b = np.zeros(X.shape[1]), 0.0
    sw = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))   # class balance
    for _ in range(iters):            # Newton steps (small problems)
        z = X @ w + b
        pr = 1 / (1 + np.exp(-z))
        g = X.T @ (sw * (pr - y)) / len(y) + l2 * w / len(y)
        gb = (sw * (pr - y)).mean()
        H = (X * (sw * pr * (1 - pr))[:, None]).T @ X / len(y) + l2 * np.eye(X.shape[1]) / len(y)
        step = np.linalg.solve(H + 1e-6 * np.eye(len(w)), g)
        w -= step
        b -= gb / max((sw * pr * (1 - pr)).mean(), 1e-6)
        if np.abs(step).max() < 1e-6:
            break
    return w, b


def grouped_cv(X, y, groups, k=5, seed=0):
    ug = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    fold = dict(zip(ug, rng.permutation(len(ug)) % k))
    f = np.array([fold[g] for g in groups])
    pred = np.zeros(len(y))
    for i in range(k):
        tr, te = f != i, f == i
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w, b = logreg((X[tr] - mu) / sd, y[tr])
        pred[te] = ((X[te] - mu) / sd) @ w + b > 0
    rec1 = (pred[y == 1] == 1).mean()
    rec0 = (pred[y == 0] == 0).mean()
    return round(float((rec1 + rec0) / 2), 3)


def run(rows_a, rows_b, kinds=("sensor", "level", "map", "all")):
    rows = rows_a + rows_b
    y = np.array([0] * len(rows_a) + [1] * len(rows_b))
    groups = [r["group"] for r in rows]
    return {k: grouped_cv(matrix(rows, k), y, groups) for k in kinds}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--own", required=True)
    ap.add_argument("--ext", action="append", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    own = load_own(a.own)
    own_pos = [r for r in own if r["y"] == 1]
    res = {}
    for e in a.ext:
        ext = load_ext(e)
        ep, en = [r for r in ext if r["y"] == 1], [r for r in ext if r["y"] == 0]
        name = Path(e).name
        res[name] = {
            "n": {"ext_pos": len(ep), "ext_neg": len(en), "own_pos": len(own_pos), "own": len(own)},
            "ext_pos vs own_pos": run(own_pos, ep),
            "ext_pos vs ext_neg": run(en, ep),
            "ext vs own": run(own, ext),
            "median": {c: [round(float(np.median([r["s"].get(c, np.nan) for r in rr])), 3)
                           for rr in (own_pos, ep, en)] for c in SENSOR + LEVEL},
        }
        print(f"== {name}  (balanced acc., 0.5 = indistinguishable)  n={res[name]['n']}")
        for q in ("ext_pos vs own_pos", "ext_pos vs ext_neg", "ext vs own"):
            print(f"  {q:22s} " + "  ".join(f"{k}={v:.2f}" for k, v in res[name][q].items()))
        print("  medians own_pos / ext_pos / ext_neg:")
        for c, v in res[name]["median"].items():
            print(f"    {c:14s} {v}")
    if a.out:
        cs.dump(res, a.out)


if __name__ == "__main__":
    main()
