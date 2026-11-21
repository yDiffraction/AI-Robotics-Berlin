#!/usr/bin/env python3
"""Tune the live readout of a finished run on its out-of-fold predictions (no retraining).

Loads <run>/fold<k>.pt, recomputes the raw output spikes on each fold's held-out own clips and
the held-out idle-recording stream, and sweeps the host readout:
  alpha   EMA factor per 10 ms frame (score = EMA(spike1 - spike0))
  hold    alarm only after the score stayed above the threshold for `hold` consecutive frames
  thr     threshold
For each (alpha, hold) it reports the best motion recall with <= --max-fa false alarms per
minute on the idle stream and <= --max-fpr alarms on idle clips, and the median detection time
relative to the swing peak (negative = before the peak).

The chosen readout is a hyperparameter picked on out-of-fold data of the training recordings;
it is not a test result on new recordings.

  python throwdet/eval_readout.py ~/throwdet/runs/e05_all_v1_nodens
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import ThrowSNN  # noqa: E402
from train import Table, own_folds, run_clips, run_stream  # noqa: E402


def ema_np(d, alpha):
    out = np.zeros_like(d)
    s = np.zeros(d.shape[:-1])
    for t in range(d.shape[-1]):
        s = (1 - alpha) * s + alpha * d[..., t]
        out[..., t] = s
    return out


def first_alarm(sc, thr, hold, start):
    above = sc > thr
    run = 0
    for t in range(start, len(sc)):
        run = run + 1 if above[t] else 0
        if run >= hold:
            return t
    return None


def stream_alarms(sc, thr, hold):
    above = sc > thr
    n, run, last = 0, 0, -10**9
    for t in range(len(sc)):
        run = run + 1 if above[t] else 0
        if run == hold and t - last >= 100:
            n += 1
            last = t
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--table", default=str(Path.home() / "throwdet/data/table.npz"))
    ap.add_argument("--max-fa", type=float, default=1.0)
    ap.add_argument("--max-fpr", type=float, default=0.15)
    args = ap.parse_args()
    run = Path(args.run)
    try:
        targs = argparse.Namespace(**json.load(open(run / "result.json"))["args"])
    except (OSError, ValueError):   # crashed while writing result.json: arguments are on the first log line
        first = open(run / "log.txt").readline()
        targs = argparse.Namespace(**json.loads(first.split(" ", 2)[2]))
    tb = Table(args.table)
    fold, _ = own_folds(tb, targs.folds, 0)
    dev = torch.device("cuda")
    clip_d, clip_y, clip_a, streams = [], [], [], []
    clip_f, clip_k, clip_s = [], [], []
    for k in range(targs.folds):
        ck = torch.load(run / f"fold{k}.pt", map_location="cpu", weights_only=False)
        model = ThrowSNN(2, arch=ck["arch"], tau=targs.tau)
        model.load_state_dict(ck["model"])
        model.to(dev)
        idx = np.flatnonzero(fold == k)
        sp = run_clips(model, tb, idx, dev).float().numpy()
        clip_d.append(sp[..., 1] - sp[..., 0]); clip_y.append(tb.y[idx]); clip_a.append(tb.anchor[idx])
        clip_f.append(tb.fname[idx]); clip_k.append(tb.kind[idx]); clip_s.append(tb.src[idx])
        st = idx[np.char.find(tb.group[idx].astype(str), "211821") >= 0]
        st = st[np.argsort(tb.start_s[st].astype(float))]
        if len(st):
            s = run_stream(model, tb, st, dev).float().numpy()
            streams.append(s[:, 1] - s[:, 0])
        del model
    D = np.concatenate(clip_d); Y = np.concatenate(clip_y); A = np.concatenate(clip_a)
    # clip-level AUC with the training readout (alpha from training), for the comparison table
    sc0 = ema_np(D, targs.alpha)
    score = np.array([sc0[b, max((A[b] if A[b] >= 0 else 70) - targs.pre, targs.warm):(A[b] if A[b] >= 0 else 70) + targs.post + 1].max()
                      if Y[b] == 1 else sc0[b, targs.warm:].max() for b in range(len(Y))])
    pos, neg = score[Y == 1], score[Y == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    minutes = sum(len(s) for s in streams) / 6000
    warm, pre, post = targs.warm, targs.pre, targs.post
    cands = []
    for alpha in (0.05, 0.1, 0.2, 0.3):
        sc = ema_np(D, alpha)
        ss = [ema_np(s, alpha) for s in streams]
        for hold in (1, 3, 5, 8):
            for thr in np.round(np.arange(-0.1, 0.95, 0.05), 2):
                det, lat, early, fp = [], [], [], []
                for b in range(len(Y)):
                    if Y[b] == 1:
                        a = A[b] if A[b] >= 0 else 70
                        t = first_alarm(sc[b][:a + post + 1], thr, hold, max(a - pre, warm))
                        det.append(t is not None)
                        early.append(first_alarm(sc[b][:max(a - pre, warm)], thr, hold, warm) is not None)
                        if t is not None:
                            lat.append((t - a) * 10)
                    else:
                        fp.append(first_alarm(sc[b], thr, hold, warm) is not None)
                fa = sum(stream_alarms(s, thr, hold) for s in ss) / max(minutes, 1e-9)
                cands.append({"alpha": alpha, "hold": hold, "thr": float(thr), "recall": round(float(np.mean(det)), 3),
                              "idle_clip_fpr": round(float(np.mean(fp)), 3), "stream_fa_per_min": round(fa, 2),
                              "latency_ms_median": float(np.median(lat)) if lat else None,
                              "early_frac": round(float(np.mean(early)), 3)})
    rows = []
    for budget_fa, budget_fpr in ((0.5, 0.1), (1.0, 0.15), (2.0, 0.2), (4.0, 0.3)):
        ok = [c for c in cands if c["stream_fa_per_min"] <= budget_fa and c["idle_clip_fpr"] <= budget_fpr]
        if ok:
            best = max(ok, key=lambda c: (c["recall"], -c["stream_fa_per_min"]))
            rows.append({"budget_fa_per_min": budget_fa, "budget_clip_fpr": budget_fpr, **best})
    print(f"{run.name}: {len(Y)} held-out own clips ({int(Y.sum())} motion), idle stream {minutes:.1f} min, "
          f"clip AUC {auc:.3f}")
    print("| FA budget (/min, clip FPR) | alpha | hold | thr | recall | idle-clip FPR | FA/min stream | median detection vs peak | on >250 ms early |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in rows:
        print(f"| {b['budget_fa_per_min']}, {b['budget_clip_fpr']} | {b['alpha']} | {b['hold']} | {b['thr']:.2f} | {b['recall']:.3f} | "
              f"{b['idle_clip_fpr']:.3f} | {b['stream_fa_per_min']:.2f} | {b['latency_ms_median']} ms | {b['early_frac']:.2f} |")
    # per recording / kind at the 2 FA/min operating point
    import re
    from collections import defaultdict
    F = np.concatenate(clip_f); Kd = np.concatenate(clip_k); S = np.concatenate(clip_s)
    op = next((r for r in rows if r["budget_fa_per_min"] == 2.0), None)
    by_rec = {}
    if op:
        sc = ema_np(D, op["alpha"])
        hits = defaultdict(list)
        for b in range(len(Y)):
            m = re.search(r"-d(\d\d_\d{6})-", str(F[b]))
            rec = (m.group(1) if m else "?") + ("/pause" if "pause" in str(S[b]) else "") + f"/{Kd[b]}"
            if Y[b] == 1:
                a = A[b] if A[b] >= 0 else 70
                hits[rec].append(first_alarm(sc[b][:a + post + 1], op["thr"], op["hold"], max(a - pre, warm)) is not None)
            else:
                hits[rec].append(first_alarm(sc[b], op["thr"], op["hold"], warm) is not None)
        by_rec = {k: [len(v), round(float(np.mean(v)), 2)] for k, v in sorted(hits.items())}
        print("alarm rate per recording/kind at the 2 FA/min operating point (motion: want high, idle: want low):")
        for k, v in by_rec.items():
            print(f"  {k:28s} n={v[0]:4d}  {v[1]:.2f}")
    json.dump({"auc": auc, "by_rec": by_rec, "rows": rows}, open(run / "readout_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
