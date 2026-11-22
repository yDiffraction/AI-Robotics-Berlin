"""Pretrain the FPGA-exportable SNN backbone on DVS Gesture (11 classes, arm/hand motion).

Input matches the board: binary 2x80x80 frames, 10 ms each, T frames per window.
Train on random T-frame windows, evaluate on fixed windows of the held-out users (24-29).

usage: python training/pretrain/train_dvsgesture.py [--epochs 30] [--max-minutes 30]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import speedpatch  # noqa: F401  (removes a per-timestep GPU sync in riverops PLIF)
from model import ThrowSNN, rate_readout

HW = 80
CELLS = 2 * HW * HW
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"


class EventWindows:
    """In-memory sparse binary frames; materialises [B, T, 2, 80, 80] batches."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.codes, self.offsets = z["codes"], z["offsets"]
        self.labels, self.users, self.n_frames = z["labels"], z["users"], z["n_frames"]

    def __len__(self):
        return len(self.labels)

    def window_idx(self, i: int, f0: int, T: int) -> np.ndarray:
        c = self.codes[self.offsets[i]:self.offsets[i + 1]]
        lo, hi = np.searchsorted(c, [f0 * CELLS, (f0 + T) * CELLS])
        return c[lo:hi].astype(np.int64) - f0 * CELLS

    def batch(self, items, T, rng=None, drop=0.0, noise=0.0) -> torch.Tensor:
        buf = np.zeros((len(items), T * CELLS), np.uint8)
        for b, (i, f0) in enumerate(items):
            idx = self.window_idx(i, f0, T)
            if rng is not None and drop > 0:
                idx = idx[rng.random(len(idx)) >= drop]
            buf[b, idx] = 1
            if rng is not None and noise > 0:
                buf[b, rng.integers(0, T * CELLS, int(noise * T * CELLS))] = 1
        return torch.from_numpy(buf).view(len(items), T, 2, HW, HW)


def random_shift(x: torch.Tensor, max_shift: int, gen: torch.Generator) -> torch.Tensor:
    if max_shift == 0:
        return x
    B = x.shape[0]
    xp = F.pad(x.flatten(1, 2), (max_shift,) * 4).view(B, x.shape[1], 2, HW + 2 * max_shift, -1)
    s = torch.randint(0, 2 * max_shift + 1, (B, 2), generator=gen)
    return torch.stack([xp[b, :, :, s[b, 0]:s[b, 0] + HW, s[b, 1]:s[b, 1] + HW] for b in range(B)])


@torch.no_grad()
def evaluate(model, ds: EventWindows, T: int, n_win: int, bs: int, dev) -> dict:
    model.eval()
    items = []
    for i in range(len(ds)):
        last = max(ds.n_frames[i] - T, 0)
        items += [(i, int(round(s))) for s in np.linspace(0, last, n_win)]
    rates = []
    for k in range(0, len(items), bs):
        x = ds.batch(items[k:k + bs], T).to(dev, torch.float32)
        rates.append(rate_readout(model(x)).cpu())
    rates = torch.cat(rates).view(len(ds), n_win, -1)
    y = torch.from_numpy(ds.labels)
    win_acc = (rates.argmax(-1) == y[:, None]).float().mean().item()
    vote_acc = (rates.sum(1).argmax(-1) == y).float().mean().item()
    silent = (rates.sum(-1) == 0).float().mean().item()
    model.train()
    return {"win_acc": win_acc, "rec_acc": vote_acc, "silent_frac": silent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--T", type=int, default=64, help="frames per window (10 ms each)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--win-per-rec", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--max-minutes", type=float, default=35.0)
    ap.add_argument("--shift", type=int, default=6)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=5e-4)
    ap.add_argument("--out", default=str(ROOT / "checkpoints"))
    ap.add_argument("--train-npz", default=str(DATA / "dvsg_train.npz"))
    args = ap.parse_args()

    dev = "xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng, gen = np.random.default_rng(0), torch.Generator().manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr, te = EventWindows(Path(args.train_npz)), EventWindows(DATA / "dvsg_test.npz")
    print(f"device {dev} | train {len(tr)} recs, test {len(te)} recs | {vars(args)}", flush=True)

    model = ThrowSNN(num_classes=11).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(tr) * args.win_per_rec / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    cfg = {"num_classes": 11, "widths": [16, 32, 64, 64], "T": args.T, "frame_ms": 10, "hw": HW,
           "input": "binary 2-polarity frames", "dataset": "DVS Gesture (users 1-23 train, 24-29 test)"}
    best, t0, history = -1.0, time.time(), []

    for ep in range(1, args.epochs + 1):
        items = [(i, int(rng.integers(0, max(tr.n_frames[i] - args.T, 0) + 1)))
                 for i in range(len(tr)) for _ in range(args.win_per_rec)]
        rng.shuffle(items)
        tl, tc, n, te0 = 0.0, 0, 0, time.time()
        for k in range(0, len(items), args.bs):
            chunk = items[k:k + args.bs]
            x = tr.batch(chunk, args.T, rng, args.drop, args.noise)
            x = random_shift(x, args.shift, gen).to(dev, torch.float32)
            y = torch.tensor([tr.labels[i] for i, _ in chunk], device=dev)
            r = rate_readout(model(x))
            loss = F.mse_loss(r, F.one_hot(y, 11).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(chunk)
            tc += (r.argmax(1) == y).sum().item()
            n += len(chunk)
        elapsed = (time.time() - t0) / 60
        line = {"epoch": ep, "loss": round(tl / n, 4), "train_acc": round(tc / n, 3),
                "epoch_s": round(time.time() - te0, 1), "elapsed_min": round(elapsed, 1)}
        stop = elapsed > args.max_minutes
        if ep % args.eval_every == 0 or ep == args.epochs or stop:
            line.update({k: round(v, 3) for k, v in evaluate(model, te, args.T, 4, 64, dev).items()})
            ckpt = {"model": model.state_dict(), "config": cfg, "epoch": ep, "metrics": line}
            torch.save(ckpt, out / "dvsg_last.pt")
            if line["win_acc"] > best:
                best = line["win_acc"]
                torch.save(ckpt, out / "dvsg_best.pt")
                line["saved_best"] = True
        history.append(line)
        print(json.dumps(line), flush=True)
        if stop:
            print(f"time budget reached ({args.max_minutes} min), stopping", flush=True)
            break

    (out / "dvsg_history.json").write_text(json.dumps(history, indent=1))

    # Export check on the best checkpoint: fused conv stack -> River Model Graph JSON.
    from riverops.export import export_for_river
    from riverops.nn import fuse_model
    model.load_state_dict(torch.load(out / "dvsg_best.pt", map_location=dev)["model"])
    stack = fuse_model(model.eval().export_module().cpu()).eval()
    (out / "dvsg_best_river_graph.json").write_text(export_for_river(stack, torch.rand(1, 4, 2, HW, HW)))
    print(f"done. best test window acc {best:.3f}; exported river graph", flush=True)


if __name__ == "__main__":
    main()
