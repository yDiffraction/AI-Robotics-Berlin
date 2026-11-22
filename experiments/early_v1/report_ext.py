#!/usr/bin/env python3
"""Tabulate early_cv.py results per run at fixed thresholds, with counts on our own clips.

  python report_ext.py results_ext/runs_ext [tag ...]

A positive counts as detected when the EMA readout crosses theta between 800 ms before and
300 ms after release AND did not already cross during the 3 s context before it ("early-fire",
which is a false alarm in the context). "ordinary sessions" = own sessions except 215305
(labels from key presses) and d01 (ball invisible). FA = alarm onsets per minute on the 60 s
held-out negative streams (1 s refractory), mean over folds.
"""
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
tags = sys.argv[2:] or sorted(p.name for p in root.iterdir() if (p / "result_f0.json").exists())
for t in tags:
    R = [json.loads(f.read_text()) for f in sorted((root / t).glob("result_f*.json"))]
    th, rows, a = R[0]["thetas"], [p for r in R for p in r["pos"]], R[0]["args"]
    print(f"\n{t}: banks={a['banks']} pos_w_src={a.get('pos_w_src')} pos_srcs={a.get('pos_srcs')} pos_w={a['pos_w']} "
          f"w_noah={a['w_noah']} p_bed={a['p_bed']} dnorm={a['dnorm']} noise={a['noise']} hot={a['hot']} folds={len(R)}")
    for tv in (0.8, 0.9, 0.95, 0.99):
        j = th.index(tv)

        def hits(sel):
            rr = [p for p in rows if sel(p)]
            h = [p for p in rr if p["first_cross"][j] is not None and p["pre_max"] < tv]
            return len(h), len(rr), [p["first_cross"][j] for p in h]
        ht, nt, lt = hits(lambda p: p["src"] == "noah" and p["sub"] == "throw")
        hf, nf, lf = hits(lambda p: p["src"] == "noah" and p["sub"] != "throw")
        ho, no, _ = hits(lambda p: p["src"] == "noah" and "215305" not in p["group"] and "d01" not in p["group"])
        hd, nd, _ = hits(lambda p: p["src"] == "dd")
        early = sum(1 for p in rows if p["src"] == "noah" and p["pre_max"] >= tv)
        fa = {s: sum(r["neg"][s]["fa_per_min"][j] for r in R if s in r["neg"]) / len(R)
              for s in ["noah", "mb", "dd", "dvsg", "mixed"]}
        lat = st.median(lt + lf) if lt + lf else float("nan")
        print(f"  th={tv}: throws {ht}/{nt} feints {hf}/{nf} | ordinary sessions {ho}/{no} | dd {hd}/{nd} | "
              f"median lead {lat:.0f} ms | early-fire {early} | FA/min noah {fa['noah']:.1f} mb {fa['mb']:.1f} "
              f"dd {fa['dd']:.1f} dvsg {fa['dvsg']:.1f} mixed {fa['mixed']:.1f}")
