#!/usr/bin/env python3
"""Whole recordings -> bit-packed binary frames [N, 1600] uint8 (2x80x80 per 10 ms), identical binning to
tools/ball_live.frames_from_file (the live path).  Also writes labels.json (events per recording)."""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
# same constants as tools/ball_live.py on the noah branch (that file is not on this branch)
SENSOR, GRID, FRAME_US = 320, 80, 10_000
REC = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])     # record_tool .bin (4-byte header)

OUT = Path(__file__).resolve().parent / "frames"
RECS = sorted(glob.glob(str(ROOT / "dataset/0[234]/samples/*.bin")))


def pack(path):
    raw = Path(path).read_bytes()
    ev = np.frombuffer(raw[4:4 + (len(raw) - 4) // REC.itemsize * REC.itemsize], dtype=REC)
    t = ev["t"].astype(np.int64)
    n = (t - int(t.min())) // FRAME_US
    cell = (np.minimum(ev["p"], 1).astype(np.int64) * GRID + np.minimum(ev["y"], SENSOR - 1) * GRID // SENSOR) * GRID \
        + np.minimum(ev["x"], SENSOR - 1) * GRID // SENSOR
    N = int(n.max()) + 1
    out = np.zeros((N, 1600), np.uint8)
    C = 4000
    for s in range(0, N, C):
        m = (n >= s) & (n < s + C)
        b = np.zeros((min(C, N - s), 12800), bool)
        b[n[m] - s, cell[m]] = True
        out[s:s + len(b)] = np.packbits(b, axis=1)
    return out


def main():
    rel = {}   # (source, round(decision,2)) -> release_ms
    for r in csv.DictReader(open(ROOT / "dataset/_dataset01_01020304/labels.csv")):
        if r["release_ms"]:
            rel[(r["source"], round(float(r["decision_s"]), 2))] = float(r["release_ms"])
    labels = {}
    for p in glob.glob(str(ROOT / "dataset/0[234]/samples/labels.csv")):
        for r in csv.DictReader(open(p)):
            d = float(r["decision_s"])
            e = {"label": r["label"], "t": d, "note": r.get("note", "")}
            k = (r["file"], round(d, 2))
            if k in rel:
                e["release_s"] = round(d + (rel[k] - 700) / 1000, 3)
            labels.setdefault(r["file"], []).append(e)
    OUT.mkdir(exist_ok=True)
    meta = {}
    for rec in RECS:
        name = Path(rec).name
        f = OUT / (Path(rec).stem + ".npy")
        if not f.exists():
            np.save(f, pack(rec))
        N = len(np.load(f, mmap_mode="r"))
        ev = sorted(labels.get(name, []), key=lambda e: e["t"])
        end = min([e["t"] for e in ev if e["label"] == "end"] + [N / 100])
        meta[name] = {"frames": N, "end_s": end, "events": ev}
        c = {}
        for e in ev:
            c[e["label"]] = c.get(e["label"], 0) + 1
        print(name, N / 100, "s end", end, c, "release", sum("release_s" in e for e in ev), flush=True)
    (OUT.parent / "labels.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
