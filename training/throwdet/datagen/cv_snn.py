#!/usr/bin/env python3
"""Grouped k-fold cross-validation of TinySNN on the REAL recordings.

The real set is tiny, so a single val/test split is noise. Here every real recording is
held out exactly once: fold k tests on the real windows of its recordings and trains on
all other real windows plus every generated clip that does NOT derive from a held-out
recording (aug/splice parents and splice donors are checked via manifest.csv). No model
selection on the test fold: the model after the last epoch is evaluated.

Reports pooled accuracy vs time relative to the decision moment (release_us), the
throw/feint-only accuracy and the confusion matrix, per source configuration.

  uv run training/cv_snn.py --data dataset/build --sources aug_train,splice_train,synth_train --folds 5
  uv run training/cv_snn.py --data dataset/build --sources synth_train --no-real-train   # sim -> real transfer
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_snn import TinySNN, bin_events, run  # noqa: E402

REAL = ("real_train", "real_val", "real_test")


def read_manifest(data):
    rows = {}
    for path in [Path(data) / "manifest.csv", *Path(data).glob("*/manifest.csv")]:
        for r in csv.DictReader(open(path)):
            folder = r.get("folder") or path.parent.name
            rows[(folder, r["file"])] = r
    return rows


def load(data, folder, names, args, cache):
    key = cache / f"cv-{Path(data).name}-{folder}-{args.steps}-{args.bin_ms}-{args.size}-{len(names)}.pt"
    if key.exists():
        return torch.load(key)
    x = torch.stack([bin_events(Path(data) / folder / n, args.steps, int(args.bin_ms * 1000), args.size).half()
                     for n in names])
    torch.save(x, key)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="dataset/build")
    p.add_argument("--classes", default="throw,feint,idle")
    p.add_argument("--sources", default="aug_train,splice_train,synth_train", help="generated training folders")
    p.add_argument("--no-real-train", action="store_true", help="train only on generated data (transfer test)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--bin-ms", type=float, default=10.0)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--shift", type=int, default=2)
    p.add_argument("--max-per-source", type=int, default=3000)
    p.add_argument("--out", default="runs/cv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    classes = args.classes.split(",")
    cid = {c: i for i, c in enumerate(classes)}
    manifest = read_manifest(args.data)
    cache = Path("runs/cache"); cache.mkdir(parents=True, exist_ok=True)

    def folder_items(folder, limit=None):
        items = [(f, r) for (fo, f), r in manifest.items() if fo == folder and r["label"] in cid]
        items.sort(key=lambda it: it[0])
        if limit and len(items) > limit:
            idx = np.sort(rng.choice(len(items), limit, replace=False))
            items = [items[i] for i in idx]
        return items

    real = [(fo, f, r) for fo in REAL for f, r in folder_items(fo)]
    parents = sorted({r["parent"] for _, _, r in real})
    order = list(parents); rng.shuffle(order)
    fold_of = {pa: i % args.folds for i, pa in enumerate(order)}

    xs, ys, meta = [], [], []
    for fo in REAL:
        items = folder_items(fo)
        if items:
            xs.append(load(args.data, fo, [f for f, _ in items], args, cache))
            for f, r in items:
                ys.append(cid[r["label"]]); meta.append(("real", r["parent"], r.get("donor", ""), r["release_us"]))
    for fo in args.sources.split(","):
        items = folder_items(fo, args.max_per_source)
        if not items:
            continue
        xs.append(load(args.data, fo, [f for f, _ in items], args, cache))
        for f, r in items:
            ys.append(cid[r["label"]]); meta.append((fo, r.get("parent", ""), r.get("donor", ""), r["release_us"]))
        print(f"{fo}: {len(items)} clips")
    X = torch.cat(xs); Y = torch.tensor(ys)
    print(f"total {len(X)} clips, {len(real)} real windows from {len(parents)} recordings, {args.folds} folds")

    t_steps = X.shape[1]
    pooled_correct = np.zeros(t_steps); pooled_n = 0
    tf_correct = np.zeros(t_steps); tf_n = 0
    cm = np.zeros((len(classes), len(classes)), int)
    for k in range(args.folds):
        held = {pa for pa, f in fold_of.items() if f == k}
        test = [i for i, m in enumerate(meta) if m[0] == "real" and m[1] in held]
        train = [i for i, m in enumerate(meta)
                 if m[1] not in held and m[2] not in held and not (m[0] == "real" and args.no_real_train)]
        xtr, ytr = X[train].float(), Y[train]
        counts = torch.bincount(ytr, minlength=len(classes)).float().clamp(min=1)
        args.class_weight = (counts.sum() / (len(classes) * counts)).to(device)
        model = TinySNN(len(classes), args.size).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(args.epochs):
            loss, acc = run(model, xtr, ytr, args, device, opt)
        xte, yte = X[test].float(), Y[test]
        model.eval()
        with torch.no_grad():
            from train_snn import running_logits
            logits = running_logits(model(xte.to(device)), args.temperature).cpu()
        pred = logits.argmax(-1)                          # [N, T]
        hit = (pred == yte[:, None]).float().numpy()
        pooled_correct += hit.sum(0); pooled_n += len(test)
        tfm = (yte != cid.get("idle", -1)).numpy()
        tf_correct += hit[tfm].sum(0); tf_n += int(tfm.sum())
        for t_, p_ in zip(yte.tolist(), pred[:, -1].tolist()):
            cm[t_, p_] += 1
        print(f"fold {k}: train {len(train)} (last-epoch train acc {acc[-1]:.1%}), test {len(test)} real windows, "
              f"acc@end {hit[:, -1].mean():.1%}")

    acc_t = pooled_correct / max(pooled_n, 1)
    tf_t = tf_correct / max(tf_n, 1)
    marks = {f"{(i + 1) * args.bin_ms - 700:+.0f}ms": (round(acc_t[i], 3), round(tf_t[i], 3))
             for i in range(t_steps) if (i + 1) % 10 == 0}
    print("accuracy (all, throw/feint only) vs time relative to decision:", marks)
    print("confusion at window end (rows = true):", cm.tolist())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cv_report.json").write_text(json.dumps({"sources": args.sources, "no_real_train": args.no_real_train,
                                                    "acc_vs_time": marks, "confusion": cm.tolist(),
                                                    "n_real": pooled_n}, indent=2))


if __name__ == "__main__":
    main()
