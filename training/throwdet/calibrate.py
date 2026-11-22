#!/usr/bin/env python3
"""Calibrate the alarm threshold of a final model (trained on all our data) to its CV run.

The readout of a CV run (alpha, hold, threshold) is validated out of fold, but a newly trained
final model has a slightly different score scale. Held-out data that NO model has seen: the
DailyDVS test persons. For the CV fold models at their readout, measure the alarm rate on those
persons' no-throw clips (arm swings + idle); then pick the final model's threshold (same alpha
and hold) that gives the same alarm rate. Writes readout_cal*.json next to the given readouts.

  python throwdet/calibrate.py ~/throwdet/deliver/A_neg e10_nodens_negheavy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_readout import ema_np, first_alarm  # noqa: E402
from model import ThrowSNN  # noqa: E402
from train import DD_HELDOUT, Table, run_clips  # noqa: E402


def neg_scores(model_path, tb, idx, dev):
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    m = ThrowSNN(2, arch=ck["arch"], tau=ck["args"]["tau"])
    m.load_state_dict(ck["model"])
    sp = run_clips(m.to(dev), tb, idx, dev).float().numpy()
    return sp[..., 1] - sp[..., 0]


def alarm_rate(d, alpha, hold, thr, warm=15):
    sc = ema_np(d, alpha)
    return float(np.mean([first_alarm(s, thr, hold, warm) is not None for s in sc]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deliver_dir")
    ap.add_argument("cv_run")
    ap.add_argument("--table", default=str(Path.home() / "throwdet/data/table.npz"))
    ap.add_argument("--runs", default=str(Path.home() / "throwdet/runs"))
    args = ap.parse_args()
    out = Path(args.deliver_dir)
    tb = Table(args.table)
    person = np.array([g.split(":", 1)[1] if g.startswith("dd:") else "" for g in tb.group.astype(str)])
    idx = np.flatnonzero((tb.domain == "dd") & np.isin(person, list(DD_HELDOUT)) & (tb.y == 0))
    dev = torch.device("cuda")
    ref = [neg_scores(Path(args.runs) / args.cv_run / f"fold{k}.pt", tb, idx, dev) for k in range(4)]
    fin = neg_scores(out / "model.pt", tb, idx, dev)
    report = {}
    for f in sorted(out.glob("readout_fa*.json")):
        r = json.load(open(f))
        a, h, t = r["alpha"], r["hold_frames"], r["threshold"]
        target = float(np.mean([alarm_rate(d, a, h, t) for d in ref]))
        grid = np.round(np.arange(-0.3, 0.96, 0.01), 2)
        rates = np.array([alarm_rate(fin, a, h, g) for g in grid])
        best = float(grid[int(np.argmin(np.abs(rates - target) + 1e-4 * np.abs(grid - t)))])
        r2 = dict(r, threshold=best, calibration={
            "method": "match the CV models' alarm rate on unseen DailyDVS no-throw clips",
            "cv_threshold": t, "target_alarm_rate": round(target, 4),
            "final_alarm_rate": round(float(rates[list(grid).index(best)]), 4),
            "final_alarm_rate_at_cv_threshold": round(alarm_rate(fin, a, h, t), 4)})
        name = f.name.replace("readout_fa", "readout_cal_fa")
        json.dump(r2, open(out / name, "w"), indent=1)
        report[f.name] = (t, best, round(target, 3))
    print(f"{out.name}: (cv thr, calibrated thr, target alarm rate) {report}")


if __name__ == "__main__":
    main()
