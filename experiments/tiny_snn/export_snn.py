#!/usr/bin/env python3
"""Export a trained TinySNN for the Kria: model JSON, weights.txt and config.yaml.

Steps: fuse BatchNorm into the convolutions, emulate fixed point (PTQ) and compare
accuracy with the float model, then write
  model.json   River Model Graph for the firmware generator on riversdk.dev
  weights.txt  integers: all weights (layer by layer), all biases, then threshold and
               leak per layer -- the layout from the Durance deployment cheatsheet
  config.yaml  spleat-server configuration (input/output shapes, scaling)

ASSUMPTIONS TO CONFIRM WITH DURANCE before flashing: weight order inside a layer is
PyTorch's [out, in, kh, kw]; weights/biases use the weight format; threshold/leak use
the potential format; `scaling` = 2**potential_frac_bits.

  uv run experiments/tiny_snn/export_snn.py runs/throw_tiny/best.pt --data recording/sessions --val 03
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from riverops.export import export_for_river
from riverops.nn import FixedPointFormat, QuantConfig, fuse_model, prepare_ptq

sys.path.insert(0, str(Path(__file__).parent))
from train_snn import TinySNN, load_split, run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--data", required=True)
    p.add_argument("--val", required=True, help="sessions used to compare float vs fixed point")
    p.add_argument("--weight-bits", type=int, default=8)
    p.add_argument("--potential-bits", type=int, default=16)
    p.add_argument("--potential-frac", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    targs = argparse.Namespace(**ckpt["args"])
    classes = ckpt["classes"]
    out = Path(args.out or Path(args.checkpoint).parent / "export")
    out.mkdir(parents=True, exist_ok=True)

    model = TinySNN(len(classes), targs.size)
    model.load_state_dict(ckpt["model"])
    model.eval()
    fused = fuse_model(model).eval()

    max_w = max(m.conv.weight.abs().max().item() for m in fused.net)
    weight_frac = args.weight_bits - 1 - max(0, math.ceil(math.log2(max_w + 1e-12)))
    wfmt = FixedPointFormat(total_bits=args.weight_bits, frac_bits=weight_frac, signed=True)
    pfmt = FixedPointFormat(total_bits=args.potential_bits, frac_bits=args.potential_frac, signed=True)
    qcfg = QuantConfig(weight_format=wfmt, potential_format=pfmt,
                       enable_weight_quant=True, enable_potential_quant=True, qat_mode=False)
    fixed = prepare_ptq(model, qcfg, fuse=True).eval()

    x, y, _ = load_split(targs.data if args.data is None else args.data, args.val.split(","), targs, Path("runs/cache"))
    device = torch.device("cpu")
    _, acc_float = run(model, x, y, targs, device)
    _, acc_fixed = run(fixed, x, y, targs, device)
    print(f"weight format Q{args.weight_bits - weight_frac}.{weight_frac} (max |w| {max_w:.3f}), "
          f"potential Q{args.potential_bits - args.potential_frac}.{args.potential_frac}")
    print(f"val accuracy  float {acc_float[-1]:.1%}  fixed-point {acc_fixed[-1]:.1%}  "
          f"(@300ms {acc_float[29]:.1%} / {acc_fixed[29]:.1%})")

    layers = [m.export_params() for m in fixed.net]
    lines = []
    for layer in layers:
        lines += layer["weights"]
    for layer in layers:
        lines += layer["bias"]
    for layer in layers:
        lines += [layer["threshold"], layer["leak"]]
    (out / "weights.txt").write_text("\n".join(str(int(v)) for v in lines))   # no trailing newline

    example = torch.zeros(1, 4, 2, targs.size, targs.size)
    (out / "model.json").write_text(export_for_river(fused.net, example))

    (out / "config.yaml").write_text(
        "application:\n"
        '  host: "0.0.0.0"\n'
        "  port: 3333\n"
        "  weights_file: /root/weights.txt\n"
        "  max_concurrency: 40\n"
        "inputs:\n"
        "  channels: 2\n"
        f"  height: {targs.size}\n"
        f"  width: {targs.size}\n"
        f"  scaling: {float(2 ** args.potential_frac)}\n"
        "outputs:\n"
        f"  channels: {len(classes)}\n"
        "  height: 1\n"
        "  width: 1\n")
    print(f"wrote {out}/weights.txt ({len(lines)} values), model.json, config.yaml; classes {classes}")


if __name__ == "__main__":
    main()
