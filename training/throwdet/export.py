#!/usr/bin/env python3
"""Export a throw-motion detector checkpoint for the Kria (same artefacts as experiments/tiny_snn/export_snn.py).

Writes into <out>/:
  model.json    River Model Graph (BN fused into the convolutions)
  weights.txt   integers: weights (layer by layer), biases, then threshold and leak per layer
  config.yaml   spleat-server configuration (input 2x80x80, output 2x1x1, scaling)
  readout.json  host-side live rule: score = EMA(spikes[1] - spikes[0]) with alpha per 10 ms frame,
                alarm when score > threshold (refractory 1 s)
and compares float vs fixed-point (PTQ) on our own clips: agreement of the alarm decision and the
balanced accuracy of both.

Same layout assumptions as experiments/tiny_snn/export_snn.py (confirm with Durance before flashing).

  python throwdet/export.py ~/throwdet/runs/<tag>/final.pt --thr 0.1
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from riverops.export import export_for_river
from riverops.nn import FixedPointFormat, QuantConfig, fuse_model, prepare_ptq

sys.path.insert(0, str(Path(__file__).parent))
from model import ThrowSNN  # noqa: E402
from train import Table, score_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--table", default=str(Path.home() / "throwdet/data/table.npz"))
    ap.add_argument("--thr", type=float, default=None, help="alarm threshold (default: from the checkpoint)")
    ap.add_argument("--alpha", type=float, default=None, help="readout EMA factor (default: from the checkpoint)")
    ap.add_argument("--hold", type=int, default=1, help="frames the score must stay above the threshold")
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--weight-frac", type=int, default=None,
                    help="fractional bits of the weights (default: as many as the largest weight allows); "
                         "5 = Q3.5, the format of the throwdet firmware")
    ap.add_argument("--potential-bits", type=int, default=16)
    ap.add_argument("--potential-frac", type=int, default=12)
    ap.add_argument("--pad-outputs", type=int, default=None,
                    help="export with this many output neurons (e.g. 11 for the gesture firmware): outputs "
                         "0 and 1 are the detector, the others get zero weights and the lowest bias, so "
                         "they never spike")
    ap.add_argument("--n-check", type=int, default=200, help="own clips used for the float/fixed comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ck["args"]
    alpha = args.alpha if args.alpha is not None else ck["readout"]["alpha"]
    thr = args.thr if args.thr is not None else ck["readout"]["thr"]
    out = Path(args.out or Path(args.checkpoint).parent / "export")
    out.mkdir(parents=True, exist_ok=True)

    model = ThrowSNN(2, arch=ck["arch"], tau=targs["tau"])
    model.load_state_dict(ck["model"])
    model.eval()
    if args.pad_outputs:
        big = ThrowSNN(args.pad_outputs, arch=ck["arch"], tau=targs["tau"])
        sd = {k: v for k, v in ck["model"].items() if not k.startswith("head.conv")}
        big.load_state_dict(sd, strict=False)
        with torch.no_grad():
            big.head.conv.weight.zero_()
            big.head.conv.weight[:2] = ck["model"]["head.conv.weight"]
            big.head.conv.bias.fill_(-100.0)  # clamped to the most negative representable value
            big.head.conv.bias[:2] = ck["model"]["head.conv.bias"]
        model = big.eval()
    net = model.export_module().eval()
    fused = fuse_model(net).eval()

    convs = [m.conv if hasattr(m, "conv") else m for m in fused]
    max_w = max(c.weight.abs().max().item() for c in convs)
    weight_frac = args.weight_bits - 1 - max(0, math.ceil(math.log2(max_w + 1e-12)))
    if args.weight_frac is not None:
        if args.weight_frac > weight_frac:
            n_clip = sum(int((c.weight.abs() > 2 ** (args.weight_bits - 1 - args.weight_frac)).sum()) for c in convs)
            print(f"WARNING: max |weight| {max_w:.3f} exceeds Q{args.weight_bits - args.weight_frac}."
                  f"{args.weight_frac}; {n_clip} weights are clipped (the float/fixed check below includes this)")
        weight_frac = args.weight_frac
    wfmt = FixedPointFormat(total_bits=args.weight_bits, frac_bits=weight_frac, signed=True)
    pfmt = FixedPointFormat(total_bits=args.potential_bits, frac_bits=args.potential_frac, signed=True)
    qcfg = QuantConfig(weight_format=wfmt, potential_format=pfmt,
                       enable_weight_quant=True, enable_potential_quant=True, qat_mode=False)
    fixed = prepare_ptq(net, qcfg, fuse=True).eval()

    tb = Table(args.table)
    rng = np.random.default_rng(0)
    own = np.flatnonzero(tb.domain == "own")
    idx = np.sort(rng.choice(own, min(args.n_check, len(own)), replace=False))
    res = {}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for name, m in (("float", net), ("fixed", fixed)):
            m.to(dev)
            sc = []
            for k in range(0, len(idx), 16):
                x = tb.dense(idx[k:k + 16]).float().to(dev)
                sp = m(x).flatten(2).cpu()
                sc.append(score_of(sp, alpha)[:, 15:].max(1).values)
            res[name] = torch.cat(sc).numpy()
            m.cpu()
    y = tb.y[idx]
    rep = {"weight_format": f"Q{args.weight_bits - weight_frac}.{weight_frac}", "max_abs_weight": max_w,
           "potential_format": f"Q{args.potential_bits - args.potential_frac}.{args.potential_frac}",
           "threshold": thr, "alpha": alpha, "n_check": len(idx),
           "decision_agreement": float(((res["float"] > thr) == (res["fixed"] > thr)).mean())}
    for name in res:
        s = res[name]
        rep[f"bal_acc_{name}"] = float(((s[y == 1] > thr).mean() + (s[y == 0] <= thr).mean()) / 2)
    print(json.dumps(rep, indent=1))

    layers = [m.export_params() for m in fixed]
    lines = []
    for layer in layers:
        lines += layer["weights"]
    for layer in layers:
        lines += layer["bias"]
    for layer in layers:
        lines += [layer["threshold"], layer["leak"]]
    (out / "weights.txt").write_text("\n".join(str(int(v)) for v in lines))
    (out / "model.json").write_text(export_for_river(fused, torch.zeros(1, 4, 2, 80, 80)))
    (out / "config.yaml").write_text(
        "application:\n  host: \"0.0.0.0\"\n  port: 3333\n  weights_file: /root/weights.txt\n  max_concurrency: 40\n"
        "inputs:\n  channels: 2\n  height: 80\n  width: 80\n"
        f"  scaling: {float(2 ** args.potential_frac)}\n"
        f"outputs:\n  channels: {args.pad_outputs or 2}\n  height: 1\n  width: 1\n")
    (out / "readout.json").write_text(json.dumps({
        "classes": ["none", "throw_motion"] + [f"unused_{k}" for k in range(2, args.pad_outputs or 2)],
        "frame_ms": 10, "input": "binary 2x80x80 (4x4 blocks of 320x320)",
        "score": "s_t = (1 - alpha) * s_{t-1} + alpha * (spike[1]_t - spike[0]_t)",
        "alpha": alpha, "threshold": thr, "hold_frames": args.hold,
        "rule": "alarm when score > threshold for hold_frames consecutive frames; then 1 s refractory",
        "refractory_ms": 1000, "ptq_check": rep}, indent=1))
    print(f"wrote {out}: weights.txt ({len(lines)} values), model.json, config.yaml, readout.json")


if __name__ == "__main__":
    main()
