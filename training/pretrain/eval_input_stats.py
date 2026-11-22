"""Eval 1: input statistics as the SNN sees them (binary 80x80, 10 ms frames), real vs synthetic.

Per window features (all in the preprocessed format):
  rate          active cells per frame (mean)
  rate_peak     max active cells in one frame
  rate_cv       std/mean of active cells over time
  on_frac       fraction of ON cells
  occupancy     fraction of the 80x80 pixels active at least once
  isolated      fraction of active cells with no active 8-neighbour in the same frame (noise-like)
  rep_interval  median frames between activations of the same pixel (both polarities)
  hot_px        share of activity in the 1 % most active pixels (hot-pixel / concentration)
  cy, cx, spread  activity centroid and spatial spread (pixels)
  rel_peak      frame of peak activity relative to release (throw/feint)

For each feature: Wasserstein distance real<->synth, compared with a real<->real baseline
(random half splits of the real windows vs. the other half), and a permutation p-value.

usage: python eval_input_stats.py ../data --out ../runs_throw/eval_input_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve
from scipy.stats import wasserstein_distance

HW, T, CELLS = 80, 120, 2 * 80 * 80
NAMES = ["throw", "feint", "idle"]


def frames(codes):
    x = np.zeros(T * CELLS, bool)
    x[codes.astype(np.int64)] = True
    return x.reshape(T, 2, HW, HW)


def feats(codes, release):
    x = frames(codes)
    per_frame = x.sum((1, 2, 3)).astype(float)
    m = per_frame.mean()
    pix = x.any(1)                                       # [T, H, W]
    nb = convolve(pix.astype(np.uint8), np.ones((1, 3, 3), np.uint8), mode="constant") - pix
    iso = (pix & (nb == 0)).sum() / max(pix.sum(), 1)
    cnt = pix.sum(0).ravel()
    gaps = []
    t_idx, p_idx = np.nonzero(pix.reshape(T, -1))
    order = np.lexsort((t_idx, p_idx))
    t_s, p_s = t_idx[order], p_idx[order]
    d = np.diff(t_s)[np.diff(p_s) == 0]
    yy, xx = np.nonzero(pix.sum(0))
    w = pix.sum(0)[yy, xx].astype(float)
    cy = float((yy * w).sum() / w.sum()) if w.sum() else 40.0
    cx = float((xx * w).sum() / w.sum()) if w.sum() else 40.0
    spread = float(np.sqrt(((yy - cy) ** 2 * w + (xx - cx) ** 2 * w).sum() / w.sum())) if w.sum() else 0.0
    top = np.sort(cnt)[::-1]
    return {"rate": m, "rate_peak": per_frame.max(), "rate_cv": per_frame.std() / (m + 1e-9),
            "on_frac": x[:, 1].sum() / max(x.sum(), 1), "occupancy": (cnt > 0).mean(),
            "isolated": iso, "rep_interval": float(np.median(d)) if len(d) else float(T),
            "hot_px": top[:len(top) // 100].sum() / max(top.sum(), 1),
            "cy": cy, "cx": cx, "spread": spread,
            "rel_peak": float(np.argmax(per_frame) - release) if release >= 0 else np.nan}, per_frame


def load(path):
    z = np.load(path)
    out, curves = [], []
    for i in range(len(z["labels"])):
        f, c = feats(z["codes"][z["offsets"][i]:z["offsets"][i + 1]], int(z["release_frame"][i]))
        f["y"], f["view"] = int(z["labels"][i]), str(z["view"][i])
        out.append(f)
        curves.append(c)
    return out, np.array(curves)


def compare(real, synth, key, rng, n_boot=500):
    a = np.array([r[key] for r in real], float)
    b = np.array([s[key] for s in synth], float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 4 or len(b) < 4:
        return None
    w_rs = wasserstein_distance(a, b)
    # baseline: real half vs real other half; matched test: real half vs synthetic sample of the same size
    base, matched = [], []
    for _ in range(n_boot):
        p = rng.permutation(len(a))
        h1, h2 = a[p[:len(a) // 2]], a[p[len(a) // 2:]]
        base.append(wasserstein_distance(h1, h2))
        matched.append(wasserstein_distance(h1, rng.choice(b, len(h2), replace=False)))
    # permutation test of the real-vs-synth Wasserstein distance (synth subsampled to the real size)
    obs, null = [], []
    for _ in range(n_boot):
        bs = rng.choice(b, len(a), replace=False)
        obs.append(wasserstein_distance(a, bs))
        pool = np.concatenate([a, bs])
        q = rng.permutation(len(pool))
        null.append(wasserstein_distance(pool[q[:len(a)]], pool[q[len(a):]]))
    p_val = float((np.array(null) >= np.mean(obs)).mean())
    return {"real_median": float(np.median(a)), "synth_median": float(np.median(b)),
            "ratio_synth_real": float(np.median(b) / np.median(a)) if np.median(a) else None,
            "W_real_synth": float(w_rs), "W_matched_median": float(np.median(matched)),
            "W_real_real_median": float(np.median(base)), "W_real_real_p95": float(np.percentile(base, 95)),
            "gap_x_baseline": float(np.median(matched) / max(np.median(base), 1e-12)), "perm_p": p_val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    real, rc = load(Path(args.data) / "throw_real.npz")
    synth, sc = load(Path(args.data) / "throw_synth.npz")
    keys = [k for k in real[0] if k not in ("y", "view")]
    res = {"all": {}, "per_class": {}}
    for k in keys:
        res["all"][k] = compare(real, synth, k, rng)
    for c, name in enumerate(NAMES):
        r = [f for f in real if f["y"] == c]
        s = [f for f in synth if f["y"] == c]
        res["per_class"][name] = {k: compare(r, s, k, rng, 200) for k in ("rate", "on_frac", "isolated",
                                                                          "occupancy", "rate_cv", "rel_peak")}
    for v in ("frontal", "profile"):
        s = [f for f in synth if f["view"] == v]
        res[f"synth_{v}"] = {k: compare(real, s, k, rng, 200) for k in ("rate", "occupancy", "spread", "cy", "cx")}
    # mean activity curve aligned to the window (release at ~70 frames)
    yr = np.array([f["y"] for f in real])
    ys = np.array([f["y"] for f in synth])
    res["mean_rate_curve"] = {name: {"real": rc[yr == c].mean(0).round(1).tolist(),
                                     "synth": sc[ys == c].mean(0).round(1).tolist()}
                              for c, name in enumerate(NAMES)}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"{'feature':13s} {'real':>8s} {'synth':>8s} {'s/r':>5s} {'gap/base':>8s} {'p':>5s}")
    for k, v in res["all"].items():
        if v:
            print(f"{k:13s} {v['real_median']:8.3f} {v['synth_median']:8.3f} "
                  f"{(v['ratio_synth_real'] or 0):5.2f} {v['gap_x_baseline']:8.2f} {v['perm_p']:5.3f}")
    for name, d in res["per_class"].items():
        print(name, {k: (round(v["real_median"], 3), round(v["synth_median"], 3), round(v["gap_x_baseline"], 1))
                     for k, v in d.items() if v})


if __name__ == "__main__":
    main()
