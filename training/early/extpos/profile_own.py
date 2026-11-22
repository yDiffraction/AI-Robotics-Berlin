#!/usr/bin/env python3
"""Measure our own GenX320 clips on the board grid -> own_profile.json (targets for match.py).

  python profile_own.py <dataset dir with labels.csv, e.g. dataset/_dataset01_01020304> own_profile.json

own_pos     per throw / feint clip: motion extent, centre, density, ON share (match.py samples
            one of them per output clip, so the targets follow our real distribution)
noise_rate  [2, 80, 80] probability per 10 ms that a cell fires in an empty scene, from the
            idle clips with almost no events (our sensor's noise floor including hot cells)
summary     10/50/90 % of every statistic per class (for the report and probe.py)
"""
import csv
import os
import sys

import numpy as np

import cellstats as cs


def main(root, out):
    rows, empty = {"throw": [], "feint": [], "idle": []}, []
    for r in csv.DictReader(open(os.path.join(root, "labels.csv"))):
        g = cs.to_cells(cs.read_bin(os.path.join(root, r["folder"], r["file"])))
        rel = int(r["release_us"]) // cs.FRAME_US if r["label"] != "idle" else None
        s = cs.clip_stats(g, rel)
        s.update(file=r["file"], source=r["source"])
        rows[r["label"]].append(s)
        if r["label"] == "idle" and g.reshape(len(g), -1).sum(1).mean() < 2:
            empty.append(g.mean(0))
    rate = np.mean(empty, 0) if empty else np.full((2, cs.HW, cs.HW), 1e-4)
    own_pos = [{k: v for k, v in s.items() if k in ("ext_h", "cx", "cy", "cells_mean", "on_frac", "file", "source")}
               for s in rows["throw"] + rows["feint"] if "ext_h" in s]
    prof = dict(own_pos=own_pos, noise_rate=rate.round(6).tolist(), n_empty=len(empty),
                summary={k: cs.summarize(v) for k, v in rows.items()})
    cs.dump(prof, out)
    print(f"{len(own_pos)} own positives, {len(empty)} empty idle clips for the noise map "
          f"(mean {rate.sum() / 2:.2f} noise cells per frame, hottest cell {rate.max():.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
