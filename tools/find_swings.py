#!/usr/bin/env python3
"""Find the arm swings in long recordings and list them in labels.csv, one row per swing.

A swing shows up as a short burst in the event rate. Every burst above --min-rate becomes a row
with label `?`; walking or picking up a ball can end up in the list as well. Label the rows with
tools/label_tool.py, then cut the clips with tools/build_dataset.py.

labels.csv (next to the recordings): file, label, decision_s, verified, note
  label       ? | throw | feint | idle (no swing) | invalid (unusable) | end
  decision_s  decision moment in seconds from the start of the file: the event-rate peak of the
              swing, just before the ball would leave the hand
  end         row that marks where the usable part of a recording stops (camera gets handled)

usage: python tools/find_swings.py dataset/03/samples/*.bin [--min-rate 22000]
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

import events_tool as ET

COLS = ["file", "label", "decision_s", "verified", "note"]
BIN = 0.02


def seconds(ev):
    t = ev["timestamp"].astype(np.int64)
    return (t - t[0]) / 1e6


def rate_peak(t, around, half=0.12):
    """Centre of the busiest 60 ms within +-half of `around`."""
    edges = np.arange(around - half - BIN, around + half + 2 * BIN, BIN)
    h = np.convolve(np.histogram(t, edges)[0], np.ones(3), "same")
    return float(edges[int(np.argmax(h[1:-1])) + 1] + BIN / 2)


def find(t, min_rate, min_gap=0.8):
    """Times of the event-rate bursts above `min_rate` events/s, at least `min_gap` apart."""
    rate = np.convolve(np.histogram(t, np.arange(0, t[-1] + BIN, BIN))[0], np.ones(3) / 3, "same") / BIN
    peaks = []
    for i in np.argsort(rate)[::-1]:
        if rate[i] < min_rate:
            break
        if all(abs(i - j) * BIN >= min_gap for j in peaks):
            peaks.append(i)
    return sorted(rate_peak(t, (i + 0.5) * BIN, 0.06) for i in peaks)


def decision_key(cols):
    """Dataset 01 uses the team's auto-label format, where the decision moment is called peak_s."""
    return "decision_s" if "decision_s" in cols else "peak_s"


def read_labels(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames), list(reader)


def write_labels(path, cols, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--min-rate", type=float, default=22000, help="events/s a burst has to reach")
    args = ap.parse_args()
    for rec in map(Path, args.recordings):
        try:
            t = seconds(ET.load(rec))
        except ValueError as e:  # capture still running: the length header is written at the end
            print(f"skipped, not a complete recording: {e}")
            continue
        path = rec.parent / "labels.csv"
        cols, rows = read_labels(path) if path.exists() else (COLS, [])
        key = decision_key(cols)
        known = [float(r[key]) for r in rows if r["file"] == rec.name]
        new = [d for d in find(t, args.min_rate) if all(abs(d - k) > 0.4 for k in known)]
        rows += [{"file": rec.name, "label": "?", key: f"{d:.2f}", "verified": 0} for d in new]
        rows.sort(key=lambda r: (r["file"], float(r[key])))
        write_labels(path, cols, rows)
        print(f"{rec.name}: {len(new)} new swings ({len(known)} already listed) -> {path}")


if __name__ == "__main__":
    main()
