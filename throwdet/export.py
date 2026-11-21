#!/usr/bin/env python3
"""Export a throw-motion detector checkpoint for the Kria (same artefacts as training/export_snn.py).

Writes into <out>/:
  model.json    River Model Graph (BN fused into the convolutions)
  weights.txt   integers: weights (layer by layer), biases, then threshold and leak per layer
  config.yaml   spleat-server configuration (input 2x80x80, output 2x1x1, scaling)
  readout.json  host-side live rule: score = EMA(spikes[1] - spikes[0]) with alpha per 10 ms frame,
                alarm when score > threshold (refractory 1 s)
and compares float vs fixed-point (PTQ) on our own clips: agreement of the alarm decision and the
balanced accuracy of both.

Same layout assumptions as training/export_snn.py (confirm with Durance before flashing).

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
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--potential-bits", type=int, default=16)
    ap.add_argument("--potential-frac", type=int, default=12)
    ap.add_argument("--n-check", type=int, default=200, help="own clips used for the float/fixed comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ck["args"]
    alpha = ck["readout"]["alpha"]
    thr = args.thr if args.thr is not None else ck["readout"]["thr"]
    out = Path(args.out or Path(args.checkpoint).parent / "export")
    out.mkdir(parents=True, exist_ok=True)

    model = ThrowSNN(2, arch=ck["arch"], tau=targs["tau"])
    model.load_state_dict(ck["model"])
    model.eval()
    net = model.export_module().eval()
    fused = fuse_model(net).eval()

    convs = [m.conv if hasattr(m, "conv") else m for m in fused]
    max_w = max(c.weight.abs().max().item() for c in convs)
    weight_frac = args.weight_bits - 1 - max(0, math.ceil(math.log2(max_w + 1e-12)))
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
    with torch.no_grad():
        for name, m in (("float", net), ("fixed", fixed)):
            sc = []
            for k in range(0, len(idx), 16):
                x = tb.dense(idx[k:k + 16]).float()
                sp = m(x).flatten(2)
                sc.append(score_of(sp, alpha)[:, 15:].max(1).values)
            res[name] = torch.cat(sc).numpy()
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
        "outputs:\n  channels: 2\n  height: 1\n  width: 1\n")
    (out / "readout.json").write_text(json.dumps({
        "classes": ["none", "throw_motion"], "frame_ms": 10, "input": "binary 2x80x80 (4x4 blocks of 320x320)",
        "score": "s_t = (1 - alpha) * s_{t-1} + alpha * (spike[1]_t - spike[0]_t)",
        "alpha": alpha, "threshold": thr, "refractory_ms": 1000, "ptq_check": rep}, indent=1))
    print(f"wrote {out}: weights.txt ({len(lines)} values), model.json, config.yaml, readout.json")


if __name__ == "__main__":
    main()
