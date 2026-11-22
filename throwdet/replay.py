#!/usr/bin/env python3
"""Replay a recording (.bin, any length) through a detector checkpoint, exactly like the live loop:
binary 10 ms frames (2x80x80), network state kept across the whole stream, host readout
score = EMA(spike[1] - spike[0]), alarm when score > threshold (1 s refractory).

Prints the alarm times and writes a PNG with the event rate, the score and the alarms. With
--labels (a samples/labels.csv of the noah branch) the labelled swings are drawn as well and
hits / misses / false alarms are counted (a hit = alarm within -600..+300 ms of a swing).

  python throwdet/replay.py ~/throwdet/runs/<tag>/final.pt recording.bin [--labels labels.csv]
"""
import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import ThrowSNN  # noqa: E402
from train import score_of  # noqa: E402

DT = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])


def frames_of(path, hw=80, bin_us=10_000):
    raw = Path(path).read_bytes()
    n = struct.unpack("!I", raw[:4])[0]
    ev = np.frombuffer(raw, DT, offset=4, count=n // DT.itemsize)
    t0 = int(ev["t"][0])
    f = (ev["t"].astype(np.int64) - t0) // bin_us
    keep = (ev["x"] < 320) & (ev["y"] < 320) & (ev["p"] < 2)
    F = int(f.max()) + 1
    x = np.zeros((F, 2, hw, hw), np.uint8)
    x[f[keep], ev["p"][keep], ev["y"][keep] // 4, ev["x"][keep] // 4] = 1
    return torch.from_numpy(x), t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("recording")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--hold", type=int, default=1)
    ap.add_argument("--readout", default=None, help="readout.json from export.py (alpha, threshold, hold)")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ThrowSNN(2, arch=ck["arch"], tau=ck["args"]["tau"])
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()
    alpha = args.alpha if args.alpha is not None else ck["readout"]["alpha"]
    thr = args.thr if args.thr is not None else ck["readout"]["thr"]
    hold = args.hold
    if args.readout:
        ro = json.load(open(args.readout))
        alpha, thr, hold = ro["alpha"], ro["threshold"], ro.get("hold_frames", 1)
    x, _ = frames_of(args.recording)
    out = []
    with torch.no_grad():
        for s in range(0, len(x), 1000):
            a = max(0, s - 60)
            o = model(x[a:s + 1000][None].to(args.device).float())[0].cpu()
            out.append(o[s - a:])
    spikes = torch.cat(out)
    sc = score_of(spikes[None], alpha)[0].numpy()
    above = sc > thr
    alarms, last, run = [], -10**9, 0
    for t in range(len(sc)):
        run = run + 1 if above[t] else 0
        if run == hold and t - last >= 100:
            alarms.append(t / 100)
            last = t
    print(f"{Path(args.recording).name}: {len(x) / 100:.1f} s, {len(alarms)} alarms at "
          + ", ".join(f"{a:.2f}" for a in alarms))
    swings = []
    if args.labels:
        name = Path(args.recording).name
        for r in csv.DictReader(open(args.labels)):
            when = r.get("decision_s") or r.get("peak_s")
            if r.get("file", r.get("source")) == name and r.get("label") in ("throw", "feint") and when:
                swings.append(float(when))
        hit = [any(-0.6 <= a - s <= 0.3 for a in alarms) for s in swings]
        fa = [a for a in alarms if not any(-0.6 <= a - s <= 0.3 for s in swings)]
        print(f"swings {len(swings)}, detected {sum(hit)}, false alarms {len(fa)}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(sc)) / 100
    fig, ax = plt.subplots(2, 1, figsize=(max(8, len(sc) / 300), 5), sharex=True)
    ax[0].plot(t, x.flatten(1).sum(1).numpy(), lw=0.6)
    ax[0].set_ylabel("active cells / 10 ms")
    ax[1].plot(t, sc, lw=0.8, label="score")
    ax[1].axhline(thr, color="k", ls="--", lw=0.6, label="threshold")
    for a in alarms:
        ax[1].axvline(a, color="r", lw=0.8)
    for s in swings:
        for k in (0, 1):
            ax[k].axvspan(s - 0.25, s + 0.1, color="g", alpha=0.2)
    ax[1].set_xlabel("s")
    ax[1].set_ylabel("throw-motion score")
    ax[1].legend(loc="upper right")
    ax[0].set_title(f"{Path(args.recording).name}: red = alarm, green = labelled swing")
    png = args.png or str(Path(args.recording).with_suffix(".replay.png").name)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    print("wrote", png)


if __name__ == "__main__":
    main()
