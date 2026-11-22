#!/usr/bin/env python3
"""Train a tiny all-convolutional spiking network with riverops.

Data: event recordings in the Durance dataset format, one folder per session:
  <data>/<session>/gesture_<ID>-<label>-....bin   (class = ID - 1)
Splits are by session (never mix one session across train/val/test).

Model: 4 SpikingConvBN2d layers, the last one outputs one channel per class at 1x1,
so the whole network maps onto the RIVER IP (no Linear head).

Training uses the running spike count at every timestep, so the network learns to
decide early; the report shows accuracy versus time into the clip.

Examples:
  # dry run on the Durance gesture dataset (5 classes)
  uv run training/train_snn.py --data data/motion_buddy_dataset \
      --classes sit,lay_down,hello,dance,no_action --train 01,02,03 --val 04 --test 05
  # throw vs feint
  uv run training/train_snn.py --data recording/sessions --classes throw,feint,idle \
      --train 01,02 --val 03 --test 04
"""

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from riverops.nn import SpikingConvBN2d
from torch import nn

EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
SENSOR = 320


def bin_events(path, steps, bin_us, size, start_us=0):
    """Count events into [steps, 2, size, size] frames (block average, like area interpolation)."""
    raw = Path(path).read_bytes()
    n = struct.unpack("!I", raw[:4])[0]
    if n != len(raw) - 4 or n % EVENT_DTYPE.itemsize:
        raise ValueError(f"invalid recording {path}")
    ev = np.frombuffer(raw, dtype=EVENT_DTYPE, offset=4)
    t = (ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0]) - start_us) // bin_us
    keep = (t >= 0) & (t < steps) & (ev["x"] < SENSOR) & (ev["y"] < SENSOR) & (ev["polarity"] < 2)
    block = SENSOR // size
    x = ev["x"][keep].astype(np.int64) // block
    y = ev["y"][keep].astype(np.int64) // block
    p = ev["polarity"][keep].astype(np.int64)
    idx = ((t[keep] * 2 + p) * size + y) * size + x
    counts = np.bincount(idx, minlength=steps * 2 * size * size).astype(np.float32)
    return torch.from_numpy(counts.reshape(steps, 2, size, size) / (block * block))


def load_split(data, sessions, args, cache_dir):
    xs, ys, names = [], [], []
    for s in sessions:
        cache = cache_dir / f"{Path(data).name}-{s}-{args.steps}-{args.bin_ms}-{args.size}.pt"
        if cache.exists():
            x, y, n = torch.load(cache)
        else:
            paths = sorted((Path(data) / s).glob("gesture_*.bin"))
            x = torch.stack([bin_events(p, args.steps, int(args.bin_ms * 1000), args.size) for p in paths])
            y = torch.tensor([int(p.name.split("-")[0].split("_")[1]) - 1 for p in paths])
            n = [p.name for p in paths]
            torch.save((x, y, n), cache)
        xs.append(x); ys.append(y); names += n
    return torch.cat(xs), torch.cat(ys), names


class TinySNN(nn.Module):
    """~4.6k weights for 3 classes at 32x32: 2->8->16->16->C channels, all spiking convs."""

    def __init__(self, classes, size, tau=1.0 / (1.0 - 0.93)):
        super().__init__()
        lif = {"initial_tau": tau, "learnable_leak": False, "v_threshold": 1.0, "reset_mode": "HARD"}
        final = size // 8
        self.net = nn.Sequential(
            SpikingConvBN2d(2, 8, kernel_size=5, stride=2, padding=2, bias=False, first_layer=True, **lif),
            SpikingConvBN2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, **lif),
            SpikingConvBN2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False, **lif),
            SpikingConvBN2d(16, classes, kernel_size=final, bias=False, **lif),
        )

    def forward(self, x):                      # x: [B, T, 2, H, W]
        return self.net(x).flatten(2)          # spikes: [B, T, classes]


def running_logits(spikes, temperature):
    """Running mean spike rate per class at every timestep -> logits [B, T, C]."""
    steps = torch.arange(1, spikes.shape[1] + 1, device=spikes.device).view(1, -1, 1)
    return spikes.cumsum(1) / steps * temperature


def run(model, x, y, args, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(x)) if training else torch.arange(len(x))
    total_loss, correct_t, n = 0.0, None, 0
    with torch.set_grad_enabled(training):
        for i in range(0, len(x), args.batch):
            idx = order[i:i + args.batch]
            xb = x[idx].to(device)
            if training and args.shift:
                dx, dy = torch.randint(-args.shift, args.shift + 1, (2,)).tolist()
                xb = torch.roll(xb, shifts=(dy, dx), dims=(3, 4))
            yb = y[idx].to(device)
            logits = running_logits(model(xb), args.temperature)
            t0 = args.warmup
            loss = F.cross_entropy(logits[:, t0:].reshape(-1, logits.shape[-1]),
                                   yb.repeat_interleave(logits.shape[1] - t0))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(idx)
            hit = (logits.argmax(-1) == yb[:, None]).float().sum(0).cpu()
            correct_t = hit if correct_t is None else correct_t + hit
            n += len(idx)
    return total_loss / n, correct_t / n


def confusion(model, x, y, classes, args, device):
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(x), args.batch):
            preds.append(running_logits(model(x[i:i + args.batch].to(device)), 1.0)[:, -1].argmax(-1).cpu())
    pred = torch.cat(preds)
    cm = torch.zeros(len(classes), len(classes), dtype=torch.long)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[t, p] += 1
    return cm


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--classes", required=True, help="comma-separated names in ID order (01, 02, ...)")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", default="")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--bin-ms", type=float, default=10.0)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--warmup", type=int, default=10, help="timesteps without loss at clip start")
    p.add_argument("--shift", type=int, default=2, help="random spatial shift (pixels) for augmentation")
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    classes = args.classes.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path("runs/cache")
    cache.mkdir(parents=True, exist_ok=True)

    t = time.time()
    xtr, ytr, _ = load_split(args.data, args.train.split(","), args, cache)
    xva, yva, _ = load_split(args.data, args.val.split(","), args, cache)
    print(f"data: train {len(xtr)}  val {len(xva)}  ({time.time() - t:.0f}s)  "
          f"class counts train {torch.bincount(ytr, minlength=len(classes)).tolist()}")

    model = TinySNN(len(classes), args.size).to(device)
    n_weights = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d))
    print(f"model: {n_weights} conv weights, device {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        tr_loss, tr_acc = run(model, xtr, ytr, args, device, optimizer)
        va_loss, va_acc = run(model, xva, yva, args, device)
        final = va_acc[-1].item()
        if final > best:
            best = final
            torch.save({"model": model.state_dict(), "classes": classes, "args": vars(args)}, out / "best.pt")
        print(f"epoch {epoch:2d}  train loss {tr_loss:.3f} acc {tr_acc[-1]:.1%}  "
              f"val loss {va_loss:.3f} acc@end {final:.1%} acc@300ms {va_acc[29]:.1%}  ({time.time() - t:.0f}s)")

    model.load_state_dict(torch.load(out / "best.pt")["model"])
    report = {"classes": classes, "weights": n_weights, "best_val_acc": best}
    if args.test:
        xte, yte, _ = load_split(args.data, args.test.split(","), args, cache)
        _, te_acc = run(model, xte, yte, args, device)
        report["test_acc_vs_time"] = {f"{(i + 1) * args.bin_ms:.0f}ms": round(a, 4)
                                      for i, a in enumerate(te_acc.tolist()) if (i + 1) % 10 == 0}
        report["test_confusion"] = confusion(model, xte, yte, classes, args, device).tolist()
        print("test accuracy vs time:", report["test_acc_vs_time"])
        print("test confusion (rows = true):", report["test_confusion"])
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"saved {out / 'best.pt'} and report.json")


if __name__ == "__main__":
    main()
