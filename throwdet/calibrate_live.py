#!/usr/bin/env python3
"""Pick the live readout (alpha, hold, threshold) of a model on labelled continuous recordings.

The model runs over each whole recording like the live loop (neuron state carried across frames).
Every labelled window (clip-windows / manual labels csv: source, start_us, end_us, label) counts as
detected when the EMA score of spike[1] - spike[0] stays above the threshold for `hold` frames
somewhere inside it. The readout with the best balanced accuracy (throw windows detected,
no_throw windows quiet) wins; ties go to the higher threshold. Writes readout.json next to --out.

  python throwdet/calibrate_live.py model.pt labels.csv raw/ --out deliver/D_person01/readout.json \
      [--compare ~/throwdet/deliver/D_e10_fold3/model.pt]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from live_view import EVENT_DTYPE, make_stateful  # noqa: E402
from model import ThrowSNN  # noqa: E402


def frames_of(path):
    ev = np.memmap(path, EVENT_DTYPE, "r", offset=4)
    t = ev["t"].astype(np.int64)
    t0 = int(t[0])
    f = (t - t0) // 10_000
    x = np.zeros((int(f[-1]) + 1, 2, 80, 80), np.uint8)
    k = (ev["x"] < 320) & (ev["y"] < 320) & (ev["p"] < 2)
    x[f[k], ev["p"][k], ev["y"][k] // 4, ev["x"][k] // 4] = 1
    return x, t0


def diff_of(ckpt, frames, dev, fixed=False):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = ThrowSNN(2, arch=ck["arch"], tau=ck["args"]["tau"])
    net.load_state_dict(ck["model"])
    net.eval()
    if fixed:  # what the board runs: BN fused, weights Q3.5, potentials Q4.12 (as export.py)
        from riverops.nn import FixedPointFormat, QuantConfig, prepare_ptq
        q = QuantConfig(weight_format=FixedPointFormat(total_bits=8, frac_bits=5, signed=True),
                        potential_format=FixedPointFormat(total_bits=16, frac_bits=12, signed=True),
                        enable_weight_quant=True, enable_potential_quant=True, qat_mode=False)
        seq = prepare_ptq(net.export_module(), q, fuse=True).eval()

        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = seq

            def forward(self, x):
                return self.seq(x).flatten(2)
        net = Wrap().eval()
    net.to(dev)
    make_stateful(net, torch)
    out = []
    with torch.no_grad():
        for a in range(0, len(frames), 1000):
            out.append(net(torch.from_numpy(frames[a:a + 1000])[None].to(dev).float())[0].cpu().numpy())
    sp = np.concatenate(out)
    return sp[:, 1] - sp[:, 0]


def ema(d, alpha):
    s, out = 0.0, np.empty(len(d), np.float32)
    for i, v in enumerate(d):
        s = (1 - alpha) * s + alpha * v
        out[i] = s
    return out


def held_above(sc, thr, hold):
    a = (sc > thr).astype(np.int32)
    if hold <= 1:
        return a.astype(bool)
    run = np.convolve(a, np.ones(hold, np.int32), "full")[:len(a)]
    return run >= hold


def evaluate(diffs, rows, t0s, fp_weight=1.0):
    best = None
    for alpha in (0.05, 0.1, 0.2):
        scs = {s: ema(d, alpha) for s, d in diffs.items()}
        for hold in (1, 3, 5, 8):
            for thr in np.round(np.arange(-0.2, 0.9, 0.01), 2):
                above = {s: held_above(sc, thr, hold) for s, sc in scs.items()}
                tp = fp = npos = nneg = 0
                for r in rows:
                    a = (int(r["start_us"]) - t0s[r["source"]]) // 10_000
                    b = (int(r["end_us"]) - t0s[r["source"]]) // 10_000
                    hit = bool(above[r["source"]][max(a, 0):b].any())
                    if r["label"] == "throw":
                        npos += 1
                        tp += hit
                    else:
                        nneg += 1
                        fp += hit
                ba = (tp / npos + 1 - fp / nneg) / 2
                key = (tp / npos - fp_weight * fp / nneg, thr)
                if best is None or key > best[0]:
                    best = (key, dict(alpha=alpha, hold=hold, threshold=float(thr), tp=tp, fn=npos - tp,
                                      fp=fp, tn=nneg - fp, balanced_accuracy=round(ba, 3)))
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("labels")
    ap.add_argument("raw_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="another checkpoint evaluated the same way")
    ap.add_argument("--fixed", action="store_true", help="calibrate on the fixed-point (board) network")
    ap.add_argument("--fp-weight", type=float, default=1.0,
                    help="cost of a false alarm relative to a missed throw (>1: fewer false alarms)")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.labels)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames, t0s = {}, {}
    for s in sorted({r["source"] for r in rows}):
        frames[s], t0s[s] = frames_of(Path(args.raw_dir) / s)
    res = {}
    for name, ck in (("model", args.checkpoint), ("compare", args.compare)):
        if ck:
            diffs = {s: diff_of(ck, x, dev, args.fixed) for s, x in frames.items()}
            res[name] = evaluate(diffs, rows, t0s, args.fp_weight)
            per_cam = {}
            for cam in sorted({r["camera"] for r in rows}):
                sub = [r for r in rows if r["camera"] == cam]
                per_cam[cam] = evaluate(diffs, sub, t0s, args.fp_weight)
            res[name + "_per_camera_best"] = per_cam
            print(name, res[name], flush=True)
    r = res["model"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"classes": ["none", "throw_motion"], "frame_ms": 10,
               "score": "s_t = (1 - alpha) * s_{t-1} + alpha * (spike[1]_t - spike[0]_t)",
               "alpha": r["alpha"], "threshold": r["threshold"], "hold_frames": r["hold"],
               "rule": "alarm when score > threshold for hold_frames consecutive frames; then 1 s refractory",
               "refractory_ms": 1000,
               "calibration": {"fixed_point": args.fixed, "fp_weight": args.fp_weight,
                                 "method": "best recall - fp_weight * false-alarm rate over labelled 3 s windows of the training "
                                         "recordings (in-sample, optimistic)", **res}}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
