#!/usr/bin/env python3
"""Export an early-detection checkpoint (ctx.py / ctx2.py, arch v1, 1 output) for the Kria board firmware.

Board constraint (docs/writeups/noah_ball_trigger.md): the arch-v1 firmware has exactly 11 outputs; a model is
deployed only by swapping weights.txt (78501 values), weights/bias Q2.6, potentials/threshold/leak Q4.12.
So the 1-output head is padded with zero rows to 11 outputs (channel 0 = throw motion, 1..10 never fire).

Writes <out>/weights.txt, model.json, config.yaml, readout.json, export_log.json and checks float vs
fixed point on whole recordings (output spikes and alarms of the host readout).

  CUDA_VISIBLE_DEVICES= python export_early.py --ckpt v2/runs/v2b/model_f0.pt --out v2/export/v2b_f0 --theta 0.7
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tim"))
sys.path.insert(0, str(HERE))
import speedpatch  # noqa: E402,F401
import ctx  # noqa: E402
from riverops.export import export_for_river  # noqa: E402
from riverops.nn import FixedPointFormat, QuantConfig, fuse_model, prepare_ptq  # noqa: E402
from snn_model import ThrowSNN, arch_from_config  # noqa: E402

N_OUT, N_VALUES = 11, 78501


def padded(ck):
    cfg = ck["config"]
    m1 = ThrowSNN(1, arch=arch_from_config(cfg))
    m1.load_state_dict(ck["model"])
    m = ThrowSNN(N_OUT, arch=arch_from_config(cfg))
    sd = m.state_dict()
    for k, v in m1.state_dict().items():
        if k.startswith("head.") and v.dim() >= 1 and v.shape[0] == 1 and sd[k].shape[0] == N_OUT:
            sd[k] = torch.zeros_like(sd[k])
            sd[k][:1] = v
            if k.endswith("bias"):
                sd[k][1:] = -1.0   # padded channels: negative drive, they can never fire
        else:
            sd[k] = v
    m.load_state_dict(sd)
    return m.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--readout", default="abs", choices=["abs", "slow", "cnt"])
    ap.add_argument("--check", nargs="+", default=["201703", "201622", "195808"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model, cfg = padded(ck)
    net = model.export_module().eval()
    fused = fuse_model(net).eval()
    # Q2.6 holds |w| <= 1.984. Per layer, divide weights, bias AND threshold by c = max|w| / 1.95 (c >= 1):
    # the membrane potential is linear in (w, b) and the reset is to 0, so the float spikes are unchanged.
    lim = 1.95
    scales = []
    with torch.no_grad():
        for mm in fused:
            c = max(1.0, float(mm.conv.weight.abs().max()) / lim, float(mm.conv.bias.abs().max()) / lim
                    if mm.conv.bias is not None else 1.0)
            mm.conv.weight /= c
            if mm.conv.bias is not None:
                mm.conv.bias /= c
            mm.lif.v_threshold = float(mm.lif.v_threshold) / c
            scales.append(round(c, 4))
    max_w = max(float(mm.conv.weight.abs().max()) for mm in fused)
    max_b = max(float(mm.conv.bias.abs().max()) for mm in fused if mm.conv.bias is not None)
    wfmt = FixedPointFormat(total_bits=8, frac_bits=6, signed=True)      # Q2.6
    pfmt = FixedPointFormat(total_bits=16, frac_bits=12, signed=True)    # Q4.12
    qcfg = QuantConfig(weight_format=wfmt, potential_format=pfmt, enable_weight_quant=True,
                       enable_potential_quant=True, qat_mode=False)
    fixed = prepare_ptq(fused, qcfg, fuse=False).eval()

    rep = {"ckpt": a.ckpt, "layer_scales": scales, "thresholds": [float(mm.lif.v_threshold) for mm in fused],
           "max_abs_weight_scaled": max_w, "max_abs_bias_scaled": max_b, "q26_range": 2.0 - 1 / 64,
           "taus": cfg.get("taus"), "readout": a.readout, "theta": a.theta}
    lab = ctx.labels()
    names = {ctx.rec_key(n): n for n in lab}
    rows = []
    with torch.no_grad():
        for k in a.check:
            fr = np.load(HERE / "frames" / (Path(names[k]).stem + ".npy"))
            x = ctx.unpack(torch.from_numpy(fr))[None]
            sp = {nm: m(x).flatten(2)[0, :, 0].numpy() for nm, m in (("float", net), ("scaled", fused), ("fixed", fixed))}
            assert (sp["float"] == sp["scaled"]).all(), "layer scaling changed the float spikes"
            y = {nm: ctx.readout(s, cfg.get("alpha", 0.85), "abs") if a.readout != "cnt" else None for nm, s in sp.items()}
            if a.readout == "cnt":
                import ctx2
                y = {nm: ctx2.readout(s, 0.85, "cnt") for nm, s in sp.items()}
            elif a.readout == "slow":
                y = {nm: ctx.readout(s, 0.85, "slow") for nm, s in sp.items()}
            y.pop("scaled", None)
            al = {nm: set(ctx.alarm_onsets(v, a.theta, 100).tolist()) for nm, v in y.items()}
            same = sum(any(abs(t - u) <= 5 for u in al["fixed"]) for t in al["float"])
            rows.append({"rec": k, "frames": len(fr), "spikes_float": int(sp["float"].sum()),
                         "spikes_fixed": int(sp["fixed"].sum()), "spike_agree": float((sp["float"] == sp["fixed"]).mean()),
                         "alarms_float": len(al["float"]), "alarms_fixed": len(al["fixed"]), "alarms_matched_50ms": same})
            print(rows[-1], flush=True)
    rep["check"] = rows

    layers = [mm.export_params() for mm in fixed]
    vals = []
    for L in layers:
        vals += L["weights"]
    for L in layers:
        vals += L["bias"]
    for L in layers:
        vals += [L["threshold"], L["leak"]]
    rep["n_values"] = len(vals)
    rep["n_values_ok"] = len(vals) == N_VALUES
    rep["leaks_q412"] = [L["leak"] for L in layers]
    (out / "weights.txt").write_text("\n".join(str(int(v)) for v in vals))
    (out / "model.json").write_text(export_for_river(fused, torch.zeros(1, 4, 2, 80, 80)))
    (out / "config.yaml").write_text(
        "application:\n  host: \"0.0.0.0\"\n  port: 3333\n  weights_file: /root/weights.txt\n  max_concurrency: 40\n"
        "inputs:\n  channels: 2\n  height: 80\n  width: 80\n  scaling: 4096.0\n"
        f"outputs:\n  channels: {N_OUT}\n  height: 1\n  width: 1\n")
    (out / "readout.json").write_text(json.dumps({
        "output_channel": 0, "meaning": "throw motion (throw or feint), channels 1..10 are padding",
        "frame_ms": 10, "readout": a.readout,
        "rule": {"abs": "y_t = 0.85 * y_{t-1} + 0.15 * spike_t", "slow": "y_t = 0.93 * y_{t-1} + 0.07 * spike_t",
                 "cnt": "y_t = (output spikes in the last 30 frames) / 30"}[a.readout],
        "theta": a.theta, "alarm": "upward crossing of theta, then 1 s refractory",
        "theta_status": "PROVISIONAL: chosen on held-out test sessions; recalibrate on the new session"}, indent=1))
    (out / "export_log.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "check"}, indent=1))


if __name__ == "__main__":
    main()
