#!/usr/bin/env python3
"""Pool the per-fold results of early_cv.py.

For every run: pick the operating threshold theta* = lowest threshold whose false-alarm rate
(mean over the per-source 60 s negative streams) is <= --fa per minute, then report there
  rec_*     detection rate (crossing within [-800 ms, +300 ms] of release) per positive set
  lat_*     median first-crossing time relative to release (ms, negative = before release)
  early     share of positives whose output already crossed theta in the context (> 800 ms before)
  fa_*      false alarms / min per negative source;  quiet = noise-only stream (drift check)
usage: python aggregate_early.py runs_early [--fa 1.0] [--fa-src noah]
  --fa-src noah   pick theta* on our own idle stream only (deployment criterion: label our next
                  recordings) instead of the mean over all negative sources
"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "runs_early")
fa_target = float(sys.argv[sys.argv.index("--fa") + 1]) if "--fa" in sys.argv else 1.0
NEG = ["noah", "mb", "dd", "dvsg", "hmdb", "mixed"]
POS_SETS = ["throw", "feint", "dd_pos", "hmdb_pos"]
fa_src = sys.argv[sys.argv.index("--fa-src") + 1] if "--fa-src" in sys.argv else ""


def pos_set(r):
    if r["src"] == "noah":
        return "throw" if r["sub"] == "throw" else "feint"
    return r["src"] + "_pos"         # external throw motions (dd = DailyDVS, hmdb = HMDB51 via v2e)


def stats(rows, th_i, th):
    hits = [r["first_cross"][th_i] for r in rows if r["first_cross"][th_i] is not None and r["pre_max"] < th]
    return (len(hits) / max(1, len(rows)), float(np.median(hits)) if hits else float("nan"),
            float(np.mean([r["pre_max"] >= th for r in rows])) if rows else float("nan"))


out = {}
hdr = f"{'tag':22s} {'thr*':>5s} {'rec_thr':>7s} {'rec_fei':>7s} {'rec_dd':>7s} {'rec_hm':>7s} {'lat_thr':>7s} {'lat_fei':>7s} {'early':>6s} " \
      + " ".join(f"fa_{n[:5]:>5s}" for n in NEG) + "  fa_quiet  y_q_last/first  folds"
print(hdr)
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    fs = sorted(d.glob("result_f*.json"))
    if not fs:
        continue
    R = [json.loads(f.read_text()) for f in fs]
    thetas = np.array(R[0]["thetas"])
    fa = {n: (np.mean([r["neg"][n]["fa_per_min"] for r in R if n in r["neg"]], 0)
              if any(n in r["neg"] for r in R) else np.full(len(thetas), np.nan)) for n in NEG}
    fa_mean = fa[fa_src] if fa_src else np.nanmean([fa[n] for n in NEG[:-1]], 0)
    quiet = np.mean([r["drift"]["fa_per_min"] for r in R], 0)
    ok = np.flatnonzero(fa_mean <= fa_target)
    ti = int(ok[0]) if len(ok) else len(thetas) - 1
    th = float(thetas[ti])
    rows = [p for r in R for p in r["pos"]]
    sets = {s: [p for p in rows if pos_set(p) == s] for s in POS_SETS}
    st = {s: stats(v, ti, th) for s, v in sets.items()}
    early = float(np.mean([p["pre_max"] >= th for p in sets["throw"] + sets["feint"]]))
    yq = (np.mean([r["drift"]["y_last10s"] for r in R]), np.mean([r["drift"]["y_first10s"] for r in R]))
    s = {"theta": th, "fa_target": fa_target, "folds": len(fs), "early": early,
         **{f"rec_{k}": v[0] for k, v in st.items()}, **{f"lat_{k}": v[1] for k, v in st.items()},
         **{f"fa_{n}": float(fa[n][ti]) for n in NEG}, "fa_quiet": float(quiet[ti]),
         "y_quiet_last": float(yq[0]), "y_quiet_first": float(yq[1]),
         "n": {k: len(v) for k, v in sets.items()},
         "curve": {"theta": thetas.tolist(), "fa_mean": fa_mean.tolist(),
                   "rec_throw": [stats(sets["throw"], i, t)[0] for i, t in enumerate(thetas)],
                   "rec_feint": [stats(sets["feint"], i, t)[0] for i, t in enumerate(thetas)],
                   "rec_dd": [stats(sets["dd_pos"], i, t)[0] for i, t in enumerate(thetas)],
                   "rec_hmdb": [stats(sets["hmdb_pos"], i, t)[0] for i, t in enumerate(thetas)]},
         "groups": {g: stats([p for p in rows if p["group"] == g], ti, th)[0]
                    for g in sorted({p["group"] for p in rows if p["src"] == "noah"})}}
    out[d.name] = s
    print(f"{d.name:22s} {th:5.2f} {s['rec_throw']:7.2f} {s['rec_feint']:7.2f} {s['rec_dd_pos']:7.2f} {s['rec_hmdb_pos']:7.2f} "
          f"{s['lat_throw']:7.0f} {s['lat_feint']:7.0f} {early:6.2f} " + " ".join(f"{s['fa_' + n]:8.2f}" for n in NEG)
          + f"  {s['fa_quiet']:8.2f}  {yq[0]:.4f}/{yq[1]:.4f}  {len(fs)}")
    if "--groups" in sys.argv:
        for g, v in s["groups"].items():
            print(f"    {g[:50]:50s} recall {v:.2f}")
json.dump(out, open(root / "summary.json", "w"), indent=1)
