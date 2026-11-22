#!/usr/bin/env python3
"""Shortcut probe: can a simple classifier tell a generated source apart from real data?

If simple window statistics separate e.g. synthetic feints from real feints, a network can
learn that artefact instead of the throw/feint cue. Accuracy near 50 % (balanced) is the goal.

Features (decision-aligned via manifest release_us; idle uses the window centre): event rate
in 50 ms bins from -300 to +500 ms (normalised by the window mean), hold ratio, ON fraction
before/after the decision, centroid spread and path length, isolated-event fraction.
Model: L2 logistic regression (numpy), grouped k-fold by parent recording.

  uv run tools/shortcut_probe.py --data dataset/build --a splice_train:feint --b real_train,real_val,real_test:feint
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import events_tool as ET  # noqa: E402


def features(ev, anchor):
    t = ev["timestamp"].astype(np.int64) - anchor
    edges = np.arange(-300_000, 500_001, 50_000)
    h = np.histogram(t, edges)[0].astype(float)
    rate = h / (h.mean() + 1)
    pre = ((t >= -250_000) & (t < 0)).sum() + 1
    hold = ((t >= 0) & (t < 150_000)).sum() / pre
    on = ev["polarity"] == 1
    on_pre = on[(t >= -200_000) & (t < 0)].mean() if ((t >= -200_000) & (t < 0)).any() else 0.5
    on_post = on[(t >= 0) & (t < 200_000)].mean() if ((t >= 0) & (t < 200_000)).any() else 0.5
    cx, cy = [], []
    for a in range(-300_000, 500_000, 25_000):
        m = (t >= a) & (t < a + 25_000)
        if m.sum() > 30:
            cx.append(ev["x"][m].mean()); cy.append(ev["y"][m].mean())
    cx, cy = np.array(cx or [0.0]), np.array(cy or [0.0])
    path = np.hypot(np.diff(cx), np.diff(cy)).sum() if len(cx) > 1 else 0.0
    tile = (ev["x"].astype(np.int64) // 5) * 64 + ev["y"].astype(np.int64) // 5
    _, counts = np.unique(tile, return_counts=True)
    iso = (counts == 1).sum() / max(len(ev), 1)
    return np.concatenate([rate, [hold, on_pre, on_post, cx.std(), cy.std(), path / 100, iso * 100]])


def load(data, spec, manifest):
    folders, label = spec.split(":") if ":" in spec else (spec, None)
    X, groups = [], []
    for row in manifest:
        if row["folder"] not in folders.split(",") or (label and row["label"] != label):
            continue
        ev = ET.load(Path(data) / row["folder"] / row["file"])
        if len(ev) < 100:
            continue
        anchor = int(row["release_us"]) if row["release_us"] not in ("", None) else int(ev["timestamp"].max()) // 2
        X.append(features(ev, anchor))
        groups.append(row.get("parent") or row["file"])
    return np.array(X), groups


def logreg(X, y, lam=1.0, iters=500, lr=0.1):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + lam * w / len(y))
        b -= lr * g.mean()
    return w, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/build")
    ap.add_argument("--a", required=True, help="folders[:label] of the generated source")
    ap.add_argument("--b", required=True, help="folders[:label] of the real reference")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-per-side", type=int, default=400)
    args = ap.parse_args()
    manifest = list(csv.DictReader(open(Path(args.data) / "manifest.csv")))
    for extra in Path(args.data).glob("*/manifest.csv"):     # generated folders keep their own manifest
        for r in csv.DictReader(open(extra)):
            r["folder"] = extra.parent.name
            manifest.append(r)
    rng = np.random.default_rng(0)
    rng.shuffle(manifest)
    Xa, ga = load(args.data, args.a, manifest)
    Xb, gb = load(args.data, args.b, manifest)
    Xa, ga = Xa[:args.max_per_side], ga[:args.max_per_side]
    X = np.vstack([Xa, Xb]); y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    groups = np.array(ga + gb)
    uniq = sorted(set(groups)); rng.shuffle(uniq)
    fold_of = {g: i % args.folds for i, g in enumerate(uniq)}
    folds = np.array([fold_of[g] for g in groups])
    pred = np.zeros(len(y))
    for k in range(args.folds):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        w, b = logreg((X[tr] - mu) / sd, y[tr])
        pred[te] = ((X[te] - mu) / sd @ w + b) > 0
    acc_a = (pred[y == 1] == 1).mean(); acc_b = (pred[y == 0] == 0).mean()
    bal = (acc_a + acc_b) / 2
    print(f"A={args.a} (n={len(Xa)})  vs  B={args.b} (n={len(Xb)})")
    print(f"balanced accuracy {bal:.1%}   (recall A {acc_a:.1%}, recall B {acc_b:.1%})   "
          f"-> {'SHORTCUT' if bal > 0.75 else 'ok-ish' if bal > 0.6 else 'good (close to chance)'}")


if __name__ == "__main__":
    main()
