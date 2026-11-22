#!/usr/bin/env python3
"""Early wind-up detector (Tim's ~/early/early_cv.py, one output, EMA readout) on WHOLE recordings.

early_cv.py evaluates on stitched streams (held-out clips concatenated with foreign negatives). This replays the
real, continuous recordings instead (same scene throughout, the person walks, picks the ball up, waits), so the
false-alarm rate is the one the demo will see and a scene switch cannot help the detector.

Frames: binary 80x80 x 2 polarities per 10 ms (tools/ball_live.frames_from_file, = the dataset binning).
Model: float ThrowSNN (the fixed-point / FPGA path matched the float model to ~2 % of spikes in board_check).
Readout: y = EMA(alpha) of the output spikes; an alarm is a crossing of theta, then `refractory` s dead time.
Positives = throws AND feints (new label: "a throw motion is happening"); anchor = labels.csv decision_s (swing
peak ~ release). Hit: first alarm in [anchor - pre, anchor + post]; latency = alarm - anchor (negative = early).
False alarms: alarms outside [anchor - pre_fa, anchor + post_fa] of every labelled event, per minute of that time.

  python tools/early_stream.py --ckpt runs/early_tim/runs_ext/r2_old_fa/model_f1.pt \
         --rec dataset/03/samples/sample_20261121_200828_124951.bin --out runs/early_tim/stream/x.npz
  python tools/early_stream.py --score runs/early_tim/stream/*.npz --thetas 0.6,0.7,0.8,0.9,0.95
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runs/robust_tim"))


def load_model(ckpt, dev):
    from snn_model import ThrowSNN, arch_from_config
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["config"]
    m = ThrowSNN(cfg.get("num_classes", 1), arch=arch_from_config(cfg)).to(dev)
    m.load_state_dict(ck["model"])
    return m.eval(), cfg


@torch.no_grad()
def run_recording(model, frames, dev, seg=3000, warm=300, roi=80):
    """frames [N,2,80,80] uint8 -> output spikes [N] (segments with `warm` frames of state warm-up)."""
    N = len(frames)
    out = np.zeros(N, np.float32)
    for s in range(0, N, seg):
        a = max(0, s - warm)
        x = torch.from_numpy(frames[a:s + seg].astype(np.float32))[None].to(dev)
        if roi < 80:
            x[..., roi:, :] = 0
        sp = model(x)[0, :, 0].float().cpu().numpy()
        out[s:s + seg] = sp[s - a:]
    return out


def ema(sp, alpha):
    y = np.zeros_like(sp)
    v = 0.0
    for k, s in enumerate(sp):
        v = alpha * v + (1 - alpha) * s
        y[k] = v
    return y


def alarms(y, theta, refractory_frames):
    out, until = [], -1
    for k in np.flatnonzero(y >= theta):
        if k >= until and (k == 0 or y[k - 1] < theta or k >= until):
            out.append(k)
            until = k + refractory_frames
    return np.array(out, int)


def load_events(rec):
    for p in glob.glob(str(ROOT / "dataset/0*/samples/labels.csv")):
        rows = [r for r in csv.DictReader(open(p)) if r.get("file") == rec]
        if rows:
            ev = [(r["label"], float(r["decision_s"])) for r in rows if r["label"] in ("throw", "feint")]
            bad = [float(r["decision_s"]) for r in rows if r["label"] == "invalid"]
            return ev, bad
    return [], []


def score(al_s, events, invalid, dur, pre=1.0, post=0.3, pre_fa=1.5, post_fa=1.5):
    used = np.zeros(len(al_s), bool)
    res = {"throw": [0, 0], "feint": [0, 0]}
    lat = {"throw": [], "feint": []}
    for lab, d in events:
        m = (al_s >= d - pre) & (al_s <= d + post)
        res[lab][1] += 1
        if m.any():
            res[lab][0] += 1
            lat[lab].append(float(al_s[m][0] - d))
    for _, d in events:
        used |= (al_s >= d - pre_fa) & (al_s <= d + post_fa)
    for d in invalid:
        used |= (al_s >= d - pre_fa) & (al_s <= d + post_fa)
    covered = (len(events) + len(invalid)) * (pre_fa + post_fa)
    fa_min = max(dur - covered, 1e-9) / 60
    return res, lat, int((~used).sum()), fa_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--rec", nargs="*", default=[])
    ap.add_argument("--range", default="", help="frac range of the recording to keep, e.g. 0,0.5 (held-out block)")
    ap.add_argument("--out", default="")
    ap.add_argument("--roi-rows", type=int, default=80)
    ap.add_argument("--score", nargs="*")
    ap.add_argument("--thetas", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--refractory", type=float, default=1.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    if a.score:
        files = sorted(set(f for g in a.score for f in glob.glob(g)))
        table = {}
        for th in [float(v) for v in a.thetas.split(",")]:
            tot = {"throw": [0, 0], "feint": [0, 0]}
            lat = {"throw": [], "feint": []}
            fa, fa_min = 0, 0.0
            for f in files:
                z = np.load(f, allow_pickle=True)
                alpha = a.alpha if a.alpha is not None else float(z["alpha"])
                y = ema(z["spikes"], alpha)
                al = alarms(y, th, int(a.refractory * 100)) / 100.0 + float(z["t_offset"])
                ev, bad = load_events(str(z["rec"]))
                r, l, n_fa, m = score(al, ev, bad, len(y) / 100)
                for k in tot:
                    tot[k][0] += r[k][0]; tot[k][1] += r[k][1]; lat[k] += l[k]
                fa += n_fa; fa_min += m
            row = {"throw_hit": tot["throw"], "feint_hit": tot["feint"],
                   "lat_ms_median": {k: (round(1000 * float(np.median(v))) if v else None) for k, v in lat.items()},
                   "fa_per_min": round(fa / max(fa_min, 1e-9), 2), "fa": fa, "minutes": round(fa_min, 1)}
            table[th] = row
            print(f"theta {th:4.2f}: throws {tot['throw'][0]:3d}/{tot['throw'][1]:<3d} feints {tot['feint'][0]:3d}/{tot['feint'][1]:<3d} "
                  f"latency median throw {row['lat_ms_median']['throw']} ms feint {row['lat_ms_median']['feint']} ms | "
                  f"false alarms {row['fa_per_min']:5.2f}/min ({fa} in {row['minutes']} min)")
        if a.json:
            Path(a.json).write_text(json.dumps({"files": files, "table": table}, indent=1))
        return

    from ball_live import frames_from_file
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(a.ckpt, dev)
    alpha = float(cfg.get("alpha", 0.85))
    for rec in a.rec:
        n, gen = frames_from_file(rec)
        frames = np.stack([f.astype(np.uint8) for f in gen])
        k0 = 0
        if a.range:
            lo, hi = [float(v) for v in a.range.split(",")]
            k0, k1 = int(lo * n), int(hi * n)
            frames = frames[k0:k1]
        sp = run_recording(model, frames, dev, roi=a.roi_rows)
        out = a.out if len(a.rec) == 1 else str(Path(a.out) / (Path(rec).stem + ".npz"))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, spikes=sp, rec=Path(rec).name, ckpt=str(a.ckpt), alpha=alpha, t_offset=k0 / 100.0)
        print(f"{rec}: {len(sp)} frames, {int(sp.sum())} output spikes -> {out}", flush=True)


if __name__ == "__main__":
    main()
