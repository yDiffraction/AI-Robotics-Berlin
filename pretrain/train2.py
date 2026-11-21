"""Pretraining v2: configurable shape, GPU augmentation, few-shot transfer probe.

Shape selection is done on *transfer*, not on DVS Gesture accuracy:
  --classes 0,1,2,3,4,5,6,10   pretrain on 8 gestures
  --probe-classes 7,8,9        afterwards: frozen backbone + new head trained on only
                               --probe-shots recordings/class (train users), tested on the
                               held-out users -> mimics fine-tuning on a few dozen throw clips.

usage (sweep):  python train2.py --arch mid --classes 0,1,2,3,4,5,6,10 --probe-classes 7,8,9
usage (final):  python train2.py --arch mid --epochs 120 --tag final
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import speedpatch  # noqa: F401
from model import ARCHS, ThrowSNN, arch_cost, rate_readout

HW, CELLS = 80, 2 * 80 * 80
ROOT = Path(__file__).resolve().parent
# Left/right mirror swaps these DVS Gesture labels (hand waves, arm rotations).
MIRROR = {1: 2, 2: 1, 3: 6, 6: 3, 4: 5, 5: 4}


class EventWindows:
    def __init__(self, path, classes=None):
        z = np.load(path)
        self.codes, offsets = z["codes"], z["offsets"]
        keep = np.arange(len(z["labels"])) if classes is None else np.flatnonzero(np.isin(z["labels"], classes))
        self.start, self.end = offsets[keep], offsets[keep + 1]
        self.labels, self.users, self.n_frames = z["labels"][keep], z["users"][keep], z["n_frames"][keep]

    def __len__(self):
        return len(self.labels)

    def batch(self, items, T) -> torch.Tensor:
        buf = np.zeros((len(items), T * CELLS), np.uint8)
        for b, (i, f0) in enumerate(items):
            c = self.codes[self.start[i]:self.end[i]]
            lo, hi = np.searchsorted(c, [f0 * CELLS, (f0 + T) * CELLS])
            buf[b, c[lo:hi].astype(np.int64) - f0 * CELLS] = 1
        return torch.from_numpy(buf).view(len(items), T, 2, HW, HW)


def prefetch(gen, n=3):
    q: queue.Queue = queue.Queue(n)

    def work():
        for item in gen:
            q.put(item)
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def augment(x, y, label_map, args):
    """x: [B,T,2,H,W] float on GPU. Random mirror (with label swap), scale, shift, drop, noise."""
    B, T = x.shape[:2]
    dev = x.device
    flip = torch.rand(B, device=dev) < args.flip
    scale = torch.empty(B, device=dev).uniform_(1 - args.scale, 1 + args.scale)
    shift = (torch.rand(B, 2, device=dev) * 2 - 1) * (args.shift / (HW / 2))
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = torch.where(flip, -1.0, 1.0) / scale
    theta[:, 1, 1] = 1.0 / scale
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, (B, 2 * T, HW, HW), align_corners=False)
    x = F.grid_sample(x.flatten(1, 2), grid, mode="nearest", padding_mode="zeros",
                      align_corners=False).view(B, T, 2, HW, HW)
    if args.drop > 0:
        x = x * (torch.rand_like(x) >= args.drop)
    if args.noise > 0:
        x = torch.maximum(x, (torch.rand_like(x) < args.noise).float())
    y = torch.where(flip, label_map[y], y)
    return x, y


def window_items(ds, T, n_per, rng):
    return [(i, int(rng.integers(0, max(ds.n_frames[i] - T, 0) + 1)))
            for i in range(len(ds)) for _ in range(n_per)]


@torch.no_grad()
def evaluate(model, ds, T, y_index, dev, n_win=4, bs=48):
    model.eval()
    items = [(i, int(round(s))) for i in range(len(ds))
             for s in np.linspace(0, max(ds.n_frames[i] - T, 0), n_win)]
    rates = []
    for k in range(0, len(items), bs):
        rates.append(rate_readout(model(ds.batch(items[k:k + bs], T).to(dev).float())).float().cpu())
    rates = torch.cat(rates).view(len(ds), n_win, -1)
    y = y_index[torch.from_numpy(ds.labels)]
    model.train()
    return {"win_acc": (rates.argmax(-1) == y[:, None]).float().mean().item(),
            "rec_acc": (rates.sum(1).argmax(-1) == y).float().mean().item(),
            "silent_frac": (rates.sum(-1) == 0).float().mean().item()}


def probe(model, args, dev, seeds=(0, 1, 2)):
    """Few-shot transfer: frozen backbone, new head on probe classes, K shots per class."""
    pc = args.probe_classes
    tr = EventWindows(args.data / "dvsg_train.npz", pc)
    te = EventWindows(args.data / "dvsg_test.npz", pc)
    y_index = torch.full((11,), -1, dtype=torch.long)
    y_index[torch.tensor(pc)] = torch.arange(len(pc))
    base_state = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    accs = []
    for seed in seeds:
        rng = np.random.default_rng(100 + seed)
        torch.manual_seed(100 + seed)
        shots = np.concatenate([rng.choice(np.flatnonzero(tr.labels == c), args.probe_shots, replace=False)
                                for c in pc])
        model.backbone.load_state_dict(base_state)
        model.reset_head(len(pc))
        model.backbone.eval()
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(model.head.parameters(), lr=3e-3)
        for step in range(args.probe_steps):
            idx = rng.choice(shots, args.bs)
            items = [(i, int(rng.integers(0, max(tr.n_frames[i] - args.T, 0) + 1))) for i in idx]
            x = tr.batch(items, args.T).to(dev).float()
            y = y_index[torch.from_numpy(tr.labels[idx])].to(dev)
            with torch.no_grad():
                feats = model.backbone(x)
            r = rate_readout(model.head(feats).flatten(2))
            loss = F.mse_loss(r, F.one_hot(y, len(pc)).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        m = evaluate(model, te, args.T, y_index, dev)
        accs.append(m["win_acc"])
    return {"probe_win_acc_mean": float(np.mean(accs)), "probe_win_acc_seeds": accs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mid", choices=list(ARCHS))
    ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--probe-classes", default="")
    ap.add_argument("--probe-shots", type=int, default=10)
    ap.add_argument("--probe-steps", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--win-per-rec", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--flip", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--shift", type=float, default=8)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=5e-4)
    ap.add_argument("--data", default=str(ROOT.parent / "data"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.classes = [int(c) for c in args.classes.split(",")]
    args.probe_classes = [int(c) for c in args.probe_classes.split(",")] if args.probe_classes else []
    args.data = Path(args.data)

    dev = "cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    name = args.tag or f"{args.arch}_c{len(args.classes)}"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)

    K = len(args.classes)
    y_index = torch.full((11,), -1, dtype=torch.long)
    y_index[torch.tensor(args.classes)] = torch.arange(K)
    # mirror map in index space (classes whose mirror partner is absent map to themselves)
    label_map = torch.arange(K)
    for a, b in MIRROR.items():
        if a in args.classes and b in args.classes:
            label_map[y_index[a]] = y_index[b]
    label_map = label_map.to(dev)

    tr = EventWindows(args.data / "dvsg_train.npz", args.classes)
    te = EventWindows(args.data / "dvsg_test.npz", args.classes)
    arch = ARCHS[args.arch]
    cost = arch_cost(arch, K)
    model = ThrowSNN(K, arch=arch).to(dev)
    cfg = {"num_classes": K, "arch": [list(a) for a in arch], "arch_name": args.arch,
           "classes": args.classes, "T": args.T, "frame_ms": 10, "hw": HW, "cost": cost,
           "input": "binary 2-polarity frames", "args": {k: (str(v) if isinstance(v, Path) else v)
                                                         for k, v in vars(args).items()}}
    print(json.dumps({"run": name, "device": dev, "train": len(tr), "test": len(te), **cost}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(tr) * args.win_per_rec / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    best, t0, history = -1.0, time.time(), []
    for ep in range(1, args.epochs + 1):
        items = window_items(tr, args.T, args.win_per_rec, rng)
        rng.shuffle(items)
        batches = (( tr.batch(items[k:k + args.bs], args.T),
                     torch.from_numpy(tr.labels[[i for i, _ in items[k:k + args.bs]]]))
                   for k in range(0, len(items), args.bs))
        tl = tc = n = 0
        te0 = time.time()
        for xb, yb in prefetch(batches):
            x = xb.to(dev, non_blocking=True).float()
            y = y_index[yb].to(dev)
            x, y = augment(x, y, label_map, args)
            r = rate_readout(model(x))
            loss = F.mse_loss(r, F.one_hot(y, K).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(y)
            tc += (r.argmax(1) == y).sum().item()
            n += len(y)
        elapsed = (time.time() - t0) / 60
        line = {"epoch": ep, "loss": round(tl / n, 4), "train_acc": round(tc / n, 3),
                "epoch_s": round(time.time() - te0, 1), "elapsed_min": round(elapsed, 1)}
        stop = elapsed > args.max_minutes
        if ep % args.eval_every == 0 or ep == args.epochs or stop:
            line.update({k: round(v, 4) for k, v in evaluate(model, te, args.T, y_index, dev).items()})
            ck = {"model": model.state_dict(), "config": cfg, "epoch": ep, "metrics": line}
            torch.save(ck, out / "last.pt")
            if line["win_acc"] > best:
                best = line["win_acc"]
                torch.save(ck, out / "best.pt")
                line["saved_best"] = True
        history.append(line)
        print(json.dumps(line), flush=True)
        if stop:
            print("time budget reached", flush=True)
            break

    summary = {"run": name, "arch": args.arch, **cost, "best_win_acc": best,
               "final": history[-1], "train_minutes": round((time.time() - t0) / 60, 1)}
    if args.probe_classes:
        model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
        summary.update(probe(model, args, dev))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
