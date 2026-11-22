#!/usr/bin/env python3
"""match.py output (labels.csv + .bin clips) -> sparse bank for early_cv.py.

Same format as prep_sources.py (sorted int32 codes ((frame*2+p)*80+y)*80+x, offsets, n_frames,
meta json). `amb` clips go to a separate <out>_amb.npz, because early_cv.py knows only
neg / pos / heldpos and amb clips must never be used as negatives.

  python to_bank.py data/extpos/clips_dd   banks/dd.npz   --src dd   [--pos-label heldpos]
  python to_bank.py data/extpos/clips_hmdb banks/hmdb.npz --src hmdb

Replace the old dd.npz completely (positives AND negatives from match.py): mixing matched
positives with negatives from the old importer would make the preprocessing a label cue.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import cellstats as cs

CELLS = 2 * cs.HW * cs.HW


def encode(ev):
    if len(ev) == 0:
        return np.zeros(0, np.int32), 120
    f = ev["t"].astype(np.int64) // cs.FRAME_US
    x = ev["x"].astype(np.int64) * cs.HW // cs.SENSOR
    y = ev["y"].astype(np.int64) * cs.HW // cs.SENSOR
    p = (ev["p"] > 0).astype(np.int64)
    c = np.unique(((f * 2 + p) * cs.HW + y) * cs.HW + x)
    return c.astype(np.int32), max(120, int(c.max() // CELLS) + 1)


def save(out, codes, nfr, meta):
    off = np.zeros(len(codes) + 1, np.int64)
    off[1:] = np.cumsum([len(c) for c in codes])
    np.savez(out, codes=np.concatenate(codes).astype(np.int32) if codes else np.zeros(0, np.int32),
             offsets=off, n_frames=np.array(nfr, np.int64), meta=np.array(json.dumps(meta)))
    labs = {}
    for m in meta:
        labs[m["label"]] = labs.get(m["label"], 0) + 1
    print(out, len(meta), "clips", labs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--src", required=True, help="source name in early_cv.SRCS (e.g. dd)")
    ap.add_argument("--pos-label", default="pos", choices=["pos", "heldpos"])
    a = ap.parse_args()
    banks = {"main": ([], [], []), "amb": ([], [], [])}
    for r in csv.DictReader(open(Path(a.clips) / "labels.csv")):
        c, n = encode(cs.read_bin(Path(a.clips) / r["folder"] / r["file"]))
        rel = int(r["release_us"]) // cs.FRAME_US if r["release_us"] else -1
        lab = {"pos": a.pos_label, "neg": "neg", "amb": "amb"}[r["role"]]
        m = dict(src=a.src, label=lab, sub=r["cls"], kind=r["role"], group=r["group"], rel=rel,
                 parent=r["parent"], variant=r["variant"], file=r["file"])
        b = banks["amb" if r["role"] == "amb" else "main"]
        b[0].append(c); b[1].append(n); b[2].append(m)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, *banks["main"])
    if banks["amb"][0]:
        save(out.with_name(out.stem + "_amb.npz"), *banks["amb"])


if __name__ == "__main__":
    main()
