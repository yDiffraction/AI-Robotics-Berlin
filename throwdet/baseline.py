#!/usr/bin/env python3
"""Non-spiking baselines on the same table and folds as train.py (sanity checks, not for the board).

  activity   logistic regression on the activity curve (active cells per 50 ms) + spatial spread.
             If this already scores high, "a lot of motion" separates the classes in that data
             and the SNN has to prove it learned more (see the hard-negative kinds).
  domain     can the same features tell the domain (own / dd / sim)? (expected: yes; that is why
             train.py balances domains and thins dense clips)

  python throwdet/baseline.py --table ~/throwdet/data/table.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import CELLS, HW, T, Table, own_folds  # noqa: E402


def feats(tb, i):
    c = tb.codes[tb.start[i]:tb.end[i]].astype(np.int64)
    f = c // CELLS
    rest = c % CELLS
    y = (rest % (HW * HW)) // HW
    x = rest % HW
    act = np.bincount(f // 5, minlength=T // 5)[:T // 5].astype(float)
    lvl = act.mean() + 1
    shape = act / lvl
    sx = [x[f // 10 == k].std() if (f // 10 == k).sum() > 5 else 0 for k in range(T // 10)]
    sy = [y[f // 10 == k].std() if (f // 10 == k).sum() > 5 else 0 for k in range(T // 10)]
    return np.concatenate([shape, [np.log(lvl), act.max() / lvl, np.mean(sx), np.mean(sy), np.max(sy) - np.min(sy)]])


def logreg(X, y, lam=1.0, iters=800, lr=0.2):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + lam * w / len(y)); b -= lr * g.mean()
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(Path.home() / "throwdet/data/table.npz"))
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    tb = Table(args.table)
    X = np.stack([feats(tb, i) for i in range(tb.n)])
    fold, _ = own_folds(tb, args.folds)
    out = {}
    pred = np.full(tb.n, np.nan)
    for k in range(args.folds):
        tr = (tb.domain == "own") & (fold != k)
        te = fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        w, b = logreg((X[tr] - mu) / sd, tb.y[tr])
        pred[te] = (X[te] - mu) / sd @ w + b
    m = fold >= 0
    y, s = tb.y[m], pred[m]
    tpr, tnr = (s[y == 1] > 0).mean(), (s[y == 0] <= 0).mean()
    out["activity_own_cv"] = {"bal_acc": round((tpr + tnr) / 2, 3), "tpr": round(tpr, 3), "tnr": round(tnr, 3)}
    src = tb.src[m]
    out["activity_own_cv_by_src_kind"] = {f"{a}/{b}": round(float(np.mean((s[(src == a) & (tb.kind[m] == b)] > 0))), 3)
                                          for a in sorted(set(src)) for b in sorted(set(tb.kind[m][src == a]))}
    # same features, trained on own only, applied to DailyDVS: does "motion" transfer?
    tr = tb.domain == "own"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    w, b = logreg((X[tr] - mu) / sd, tb.y[tr])
    dd = tb.domain == "dd"
    sdd = (X[dd] - mu) / sd @ w + b
    out["activity_own_to_dd_detect_rate_by_kind"] = {k: round(float((sdd[tb.kind[dd] == k] > 0).mean()), 3)
                                                     for k in sorted(set(tb.kind[dd]))}
    # domain probe (own vs dd vs sim, one-vs-rest accuracy of own)
    for d in ("dd", "sim"):
        mm = np.isin(tb.domain, ["own", d])
        yy = (tb.domain[mm] == d).astype(float)
        idx = np.random.default_rng(0).permutation(mm.sum())
        half = len(idx) // 2
        Xm = X[mm]
        mu, sd = Xm[idx[:half]].mean(0), Xm[idx[:half]].std(0) + 1e-6
        w, b = logreg((Xm[idx[:half]] - mu) / sd, yy[idx[:half]])
        ss = (Xm[idx[half:]] - mu) / sd @ w + b
        yt = yy[idx[half:]]
        out[f"domain_probe_own_vs_{d}_bal_acc"] = round(float(((ss[yt == 1] > 0).mean() + (ss[yt == 0] <= 0).mean()) / 2), 3)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
