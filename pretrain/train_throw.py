"""Fine-tune the DVS-Gesture-pretrained hires SNN on throw / feint / idle.

Data (prep_throw.py): throw_real.npz (real windows, grouped by source recording) and
throw_synth.npz (procedural synthetic clips). 1.2 s windows = 120 binary 10 ms frames.

The synthetic set is split once (seed 0): 50 clips/class are a fixed synth test set, the rest
(1050) is the synthetic training pool; --synth-n draws a class-balanced subset of it.
Real data: --real none (train on synthetic only, test on all real = TSTR), --real all (final model),
or --real cv --folds 5 --fold k (grouped by recording; the held-out recordings are the test set).

No model selection on test data: the model after the last step is evaluated.
Writes <out>/<tag>/result.json (per-window predictions) and checkpoints at --save-steps (for the
gradient analysis) plus final.pt.

usage: python train_throw.py --init ../runs/sweep_hires/best.pt --synth-n 1050 --real cv --fold 0 --tag cv0_s1050
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import speedpatch  # noqa: F401
from model import ARCHS, ThrowSNN, arch_from_config, rate_readout
from train2 import EventWindows, augment, prefetch

CLASSES = ["throw", "feint", "idle"]
K = 3
ROOT = Path(__file__).resolve().parent


class Windows(EventWindows):
    def __init__(self, path):
        super().__init__(path)
        z = np.load(path)
        self.release = z["release_frame"]
        self.parent, self.file, self.view = z["parent"], z["fname"], z["view"]


def synth_split(ds, per_class_test=50, seed=0):
    rng = np.random.default_rng(seed)
    test, pool = [], []
    for c in range(K):
        idx = rng.permutation(np.flatnonzero(ds.labels == c))
        test += idx[:per_class_test].tolist()
        pool.append(idx[per_class_test:])
    return np.array(sorted(test)), pool


def synth_subset(pool, n, seed):
    """Class-balanced, nested subsets (the first n of a fixed per-class order)."""
    rng = np.random.default_rng(1000 + seed)
    order = [rng.permutation(p) for p in pool]
    per = [n // K + (1 if c < n % K else 0) for c in range(K)]
    return np.array(sorted(np.concatenate([o[:k] for o, k in zip(order, per)])), np.int64)


def real_folds(ds, folds, seed=0):
    """Assign source recordings to folds, stratified by the recording's throw/feint label."""
    rng = np.random.default_rng(seed)
    kind = {}
    for p, y in zip(ds.parent, ds.labels):
        kind[p] = min(kind.get(p, 9), int(y))          # 0 throw, 1 feint, 2 idle-only
    fold, k = {}, 0
    for c in range(K):
        ps = sorted(p for p, v in kind.items() if v == c)
        for p in rng.permutation(ps):
            fold[p] = k % folds
            k += 1
    return np.array([fold[p] for p in ds.parent])


def time_shift(x, max_shift):
    """Shift each sample by up to +-max_shift frames (zero fill): release-time jitter."""
    if max_shift <= 0:
        return x
    out = torch.zeros_like(x)
    for b, s in enumerate(torch.randint(-max_shift, max_shift + 1, (len(x),)).tolist()):
        if s >= 0:
            out[b, s:] = x[b, :x.shape[1] - s]
        else:
            out[b, :s] = x[b, -s:]
    return out


@torch.no_grad()
def predict(model, ds, idx, T, dev, bs=16):
    """Per-window output spikes [N, T, K] (float16 on CPU) and final rates."""
    model.eval()
    outs = []
    for k in range(0, len(idx), bs):
        items = [(int(i), 0) for i in idx[k:k + bs]]
        outs.append(model(ds.batch(items, T).to(dev).float()).half().cpu())
    model.train()
    return torch.cat(outs) if outs else torch.zeros(0, T, K)


def summarize(spikes, y, release):
    """Accuracy at window end and running-count accuracy at release + 0/100/200/300 ms."""
    res = {"n": len(y)}
    if len(y) == 0:
        return res
    cum = spikes.float().cumsum(1)
    res["acc_end"] = float((cum[:, -1].argmax(-1) == y).float().mean())
    tf = y < 2
    if tf.any():
        res["acc_end_throw_feint"] = float((cum[tf, -1].argmax(-1) == y[tf]).float().mean())
    for ms in (0, 100, 200, 300):
        f = torch.where(release >= 0, release + ms // 10, torch.full_like(release, spikes.shape[1] - 1))
        f = f.clamp(0, spikes.shape[1] - 1)
        pred = cum[torch.arange(len(y)), f].argmax(-1)
        res[f"acc_rel+{ms}ms"] = float((pred == y).float().mean())
        if tf.any():
            res[f"acc_rel+{ms}ms_throw_feint"] = float((pred[tf] == y[tf]).float().mean())
    cm = torch.zeros(K, K, dtype=torch.long)
    for t, p in zip(y.tolist(), cum[:, -1].argmax(-1).tolist()):
        cm[t, p] += 1
    res["confusion_end"] = cm.tolist()
    res["silent_frac"] = float((cum[:, -1].sum(-1) == 0).float().mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="", help="pretrained checkpoint; empty = random init")
    ap.add_argument("--arch", default="hires", choices=list(ARCHS))
    ap.add_argument("--synth-n", type=int, default=1050)
    ap.add_argument("--real", default="cv", choices=["none", "all", "cv"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--real-frac", type=float, default=0.25, help="share of real windows per batch when mixing")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3, help="backbone LR (2x for random init)")
    ap.add_argument("--head-lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--flip", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--shift", type=float, default=8)
    ap.add_argument("--tshift", type=int, default=8, help="max time shift in frames")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=5e-4)
    ap.add_argument("--save-steps", default="0,100,300")
    ap.add_argument("--data", default=str(ROOT.parent / "data"))
    ap.add_argument("--out", default=str(ROOT.parent / "runs_throw"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=int, default=0, help="bf16 autocast for training steps")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    real, synth = Windows(data / "throw_real.npz"), Windows(data / "throw_synth.npz")

    s_test, pool = synth_split(synth)
    s_train = synth_subset(pool, args.synth_n, args.seed) if args.synth_n > 0 else np.zeros(0, np.int64)
    if args.real == "cv":
        fold = real_folds(real, args.folds)
        r_test, r_train = np.flatnonzero(fold == args.fold), np.flatnonzero(fold != args.fold)
    elif args.real == "all":
        r_test, r_train = np.zeros(0, np.int64), np.arange(len(real))
    else:
        r_test, r_train = np.arange(len(real)), np.zeros(0, np.int64)
    assert len(s_train) + len(r_train) > 0

    if args.init:
        ck = torch.load(args.init, map_location=dev)
        model = ThrowSNN(ck["config"]["num_classes"], arch=arch_from_config(ck["config"])).to(dev)
        model.load_state_dict(ck["model"])
        model.reset_head(K)
        arch = model.arch
    else:
        arch = ARCHS[args.arch]
        model = ThrowSNN(K, arch=arch).to(dev)
    bb_lr = args.lr * (1 if args.init else 2)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": bb_lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=args.wd)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)))
    label_map = torch.arange(K, device=dev)  # mirroring keeps throw/feint/idle
    cfg = {"num_classes": K, "classes": CLASSES, "arch": [list(a) for a in arch], "arch_name": args.arch,
           "T": args.T, "frame_ms": 10, "hw": 80, "input": "binary 2-polarity frames",
           "args": vars(args)}
    save_steps = {int(s) for s in args.save_steps.split(",") if s}
    print(json.dumps({"run": args.tag, "synth_train": len(s_train), "real_train": len(r_train),
                      "real_test": len(r_test), "synth_test": len(s_test)}), flush=True)

    n_real = 0 if len(r_train) == 0 else (args.bs if len(s_train) == 0 else round(args.bs * args.real_frac))

    def batches():
        for _ in range(args.steps):
            ri = rng.choice(r_train, n_real) if n_real else np.zeros(0, np.int64)
            si = rng.choice(s_train, args.bs - n_real) if args.bs - n_real else np.zeros(0, np.int64)
            parts = []
            if len(ri):
                parts.append((real.batch([(int(i), 0) for i in ri], args.T), real.labels[ri]))
            if len(si):
                parts.append((synth.batch([(int(i), 0) for i in si], args.T), synth.labels[si]))
            yield torch.cat([p[0] for p in parts]), torch.from_numpy(np.concatenate([p[1] for p in parts]))

    def save(name, step):
        torch.save({"model": model.state_dict(), "config": cfg, "step": step}, out / name)

    t0, run_l, run_a, log = time.time(), [], [], []
    for step, (xb, yb) in enumerate(prefetch(batches())):
        if step in save_steps:
            save(f"step{step}.pt", step)
        x = time_shift(xb.to(dev, non_blocking=True).float(), args.tshift)
        y = yb.to(dev)
        x, y = augment(x, y, label_map, args)
        with torch.autocast("cuda", torch.bfloat16, enabled=bool(args.amp)):
            r = rate_readout(model(x)).float()
        loss = F.mse_loss(r, F.one_hot(y, K).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run_l.append(loss.item())
        run_a.append((r.argmax(1) == y).float().mean().item())
        if (step + 1) % 100 == 0:
            log.append({"step": step + 1, "loss": round(float(np.mean(run_l)), 4),
                        "train_acc": round(float(np.mean(run_a)), 3),
                        "min": round((time.time() - t0) / 60, 1)})
            run_l, run_a = [], []
            print(json.dumps(log[-1]), flush=True)
    save("final.pt", args.steps)

    res = {"tag": args.tag, "args": vars(args), "train_log": log, "minutes": round((time.time() - t0) / 60, 1)}
    for name, ds, idx in (("real_test", real, r_test), ("synth_test", synth, s_test)):
        sp = predict(model, ds, idx, args.T, dev)
        y = torch.from_numpy(ds.labels[idx])
        res[name] = summarize(sp, y, torch.from_numpy(ds.release[idx]))
        res[name + "_windows"] = [{"i": int(i), "file": str(ds.file[i]), "parent": str(ds.parent[i]),
                                   "y": int(ds.labels[i]), "release_frame": int(ds.release[i]),
                                   "counts": sp[j].float().sum(0).tolist()} for j, i in enumerate(idx)]
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print("RESULT " + json.dumps({k: res[k] for k in ("tag", "real_test", "synth_test", "minutes")}), flush=True)


if __name__ == "__main__":
    main()
