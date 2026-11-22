#!/usr/bin/env python3
"""Checkpoint (ThrowSNN, arch v1) -> board folder: weights.txt, model.json, config.yaml, readout.json, model.pt.

Board format (firmware arch v1, 78501 values, one integer per line):
  all conv weights layer by layer, PyTorch order [out, in, kh, kw], Q2.6 (x64, clipped to int8)
  all biases layer by layer, Q2.6; head padded to 11 outputs, padding channels get bias -1.0 (never fire)
  per layer: threshold, leak (= 1/tau), Q4.12 (x4096)
BatchNorm is folded into the conv. If a layer's weights or bias do not fit Q2.6, the whole layer (weights, bias,
threshold) is divided by one factor; that is exact in float (the PLIF update is linear in the input current).

Reproduces live_early/model_v3/weights.txt bit for bit (md5 44f403275d8ea3c87d4e0bf93078cf08):
  python live_early/hitl/export_board.py live_early/model_v3/model.pt --out <dir> --check-against live_early/model_v3
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent
sys.path.insert(0, str(LIVE))
sys.path.insert(0, str(HERE))
from snn_model import ThrowSNN, arch_from_config  # noqa: E402

P_SCALE = 4096                       # potentials Q4.12


def board_format(model_dir: Path):
    """(outputs, weight fractional bits) of the firmware a model folder targets: config.yaml outputs, and the
    weight format in readout.json's ptq_check (throwdet exports); v3 has none -> Q2.6."""
    cfg = (model_dir / "config.yaml").read_text().split("outputs:")[1]
    n_out = int(cfg.split("channels:")[1].split()[0])
    ro = json.loads((model_dir / "readout.json").read_text())
    wf = (ro.get("ptq_check") or {}).get("weight_format", "Q2.6")
    return n_out, int(wf.split(".")[1])


def fold(layer):
    """(weight, bias, threshold, tau) of one spiking conv with BatchNorm folded in, float64."""
    w = layer.conv.weight.detach().double()
    b = layer.conv.bias.detach().double() if layer.conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    bn = getattr(layer, "bn", None)
    if bn is not None:
        g = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        w = w * g[:, None, None, None]
        b = (b - bn.running_mean.double()) * g + bn.bias.detach().double()
    tau = float(torch.as_tensor(layer.lif.tau).detach())
    return w, b, float(layer.lif.v_threshold), tau


def q(x, scale):
    return np.round(np.asarray(x, np.float64) * scale).astype(np.int64)   # round half to even, like the v3 export


def export(model, out_dir: Path, theta: float, extra_log: dict | None = None, model_dir: Path = LIVE / "model_v3"):
    """Write a board folder. model_dir is the folder whose firmware format, model.json, config.yaml and readout rule
    are kept (the model the training started from)."""
    from readout_spec import Spec
    model_dir = Path(model_dir)
    n_out, frac = board_format(model_dir)
    w_scale = 2 ** frac
    w_max = 127 / w_scale
    target = w_max * 1.95 / 1.984375                 # v3 export: max |w| after scaling = 1.95 at Q2.6
    layers = list(model.backbone) + [model.head]
    n_real = model.head.conv.out_channels
    ws, bs, tail = [], [], []
    log: dict = {"layer_scales": [], "thresholds": [], "taus": []}
    for i, L in enumerate(layers):
        w, b, th, tau = fold(L)
        head = i == len(layers) - 1
        if head and n_out > n_real:                  # pad to the firmware's outputs; padding never fires
            pad = n_out - n_real
            w = torch.cat([w, torch.zeros((pad, *w.shape[1:]), dtype=w.dtype)])
            b = torch.cat([b, torch.full((pad,), -1.0, dtype=b.dtype)])
        m = max(float(w.abs().max()), float(b[:n_real if head else None].abs().max()))
        s = 1.0 if m <= w_max else round(m / target, 4)
        w, b, th = w / s, b / s, th / s
        if head and n_out > n_real:
            b[n_real:] = -1.0
        ws.append(np.clip(q(w.numpy().ravel(), w_scale), -128, 127))
        bs.append(np.clip(q(b.numpy(), w_scale), -128, 127))
        tail += [int(q(th, P_SCALE)), int(q(1.0 / tau, P_SCALE))]
        log["layer_scales"].append(s)
        log["thresholds"].append(th)
        log["taus"].append(tau)
    values = np.concatenate(ws + bs + [np.array(tail, np.int64)])
    out_dir.mkdir(parents=True, exist_ok=True)
    # no trailing newline, exactly like the v2b / v3 exports
    (out_dir / "weights.txt").write_bytes(b"\n".join(str(int(v)).encode() for v in values))
    for name in ("model.json", "config.yaml"):
        shutil.copyfile(model_dir / name, out_dir / name)
    (out_dir / "readout.json").write_text(json.dumps(Spec(model_dir).with_theta(theta), indent=1))
    md5 = hashlib.md5((out_dir / "weights.txt").read_bytes()).hexdigest()
    ref = model_dir / "weights.txt"
    n_ref = len(ref.read_text().split()) if ref.exists() else None
    log.update(extra_log or {})
    log.update(n_values=int(values.size), n_values_ok=n_ref in (None, values.size), md5=md5, leaks_q412=tail[1::2],
               theta=theta, weight_format=f"Q{8 - frac}.{frac}", outputs=n_out, template=str(model_dir))
    if n_ref is not None and n_ref != values.size:
        raise RuntimeError(f"export has {values.size} values, {ref} has {n_ref}")
    (out_dir / "export_log.json").write_text(json.dumps(log, indent=1))
    return md5, log


def load_ckpt(path):
    """live_early checkpoints ({model, config}) and throwdet ones ({arch, args, model})."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if "config" not in ck:
        ck["config"] = {"arch": [tuple(x) for x in ck["arch"]], "num_classes": int(sd["head.conv.weight"].shape[0])}
    cfg = ck["config"]
    m = ThrowSNN(cfg.get("num_classes", int(sd["head.conv.weight"].shape[0])), arch=arch_from_config(cfg))
    m.load_state_dict(sd)
    return m.eval(), ck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--theta", type=float, default=0.85)
    ap.add_argument("--model-dir", default=str(LIVE / "model_v3"), help="template folder (firmware format, readout)")
    ap.add_argument("--check-against", help="board folder whose weights.txt must come out identical")
    a = ap.parse_args()
    model, ck = load_ckpt(a.ckpt)
    out = Path(a.out)
    md5, log = export(model, out, a.theta, model_dir=Path(a.model_dir))
    if Path(a.ckpt).resolve() != (out / "model.pt").resolve():
        torch.save(ck, out / "model.pt")
    print(json.dumps({"md5": md5, "n_values": log["n_values"], "layer_scales": log["layer_scales"]}))
    if a.check_against:
        ref = (Path(a.check_against) / "weights.txt").read_text().split()
        new = (out / "weights.txt").read_text().split()
        diff = sum(x != y for x, y in zip(ref, new)) + abs(len(ref) - len(new))
        print(f"values differing from {a.check_against}: {diff} of {len(ref)}")
        sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
