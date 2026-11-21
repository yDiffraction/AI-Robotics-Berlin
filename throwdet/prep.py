#!/usr/bin/env python3
"""Build the training table for the binary throw-motion detector.

Every clip (1.2 s, timestamps from 0) becomes binary [120, 2, 80, 80] frames of 10 ms, stored as
sorted active-cell codes (code = ((frame*2+p)*80+y)*80+x), the board's input format (despot
downsampling 4, 100 Hz). Same encoding as pretrain/prep_throw.py on the Tim branch.

Target: 1 = a throw motion is happening (real throw OR feint: the arm swings like a throw),
0 = anything else. Motion clips have their decision moment (event-rate peak of the swing) at
`anchor` frames (70 for all our own clips).

Domains (used for balanced sampling and per-domain evaluation):
  own    GenX320 recordings of the team (noah branch, dataset/_dataset01_01020304 + idle pauses
         from _dataset01_010203), plus aug/splice clips derived from them (dataset/build)
  sim    procedural synth v2 clips and phone-video conversions (dataset/build)
  dd     DailyDVS-200 imports (tools/dailydvs_import.py): throw/toss = 1, arm_no_release/idle = 0

usage: python throwdet/prep.py --noah ~/throwdet/noah/dataset --build dataset/build \
           --dd ~/ext/dailydvs --out ~/throwdet/data/table.npz
"""
from __future__ import annotations

import argparse
import csv
import re
import struct
from pathlib import Path

import numpy as np

DT = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])
SENSOR, HW, BIN_US, T = 320, 80, 10_000, 120
POS = {"throw", "feint", "toss"}


def codes_of(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = struct.unpack("!I", raw[:4])[0]
    assert n == len(raw) - 4 and n % DT.itemsize == 0, path
    ev = np.frombuffer(raw, DT, offset=4)
    t = ev["t"].astype(np.int64)
    f = t // BIN_US
    keep = (f >= 0) & (f < T) & (ev["x"] < SENSOR) & (ev["y"] < SENSOR) & (ev["p"] < 2)
    b = SENSOR // HW
    c = ((f[keep] * 2 + ev["p"][keep]) * HW + ev["y"][keep] // b) * HW + ev["x"][keep] // b
    return np.unique(c.astype(np.uint32))


def own_rows(noah: Path):
    rows = []
    d4 = noah / "_dataset01_01020304"
    for r in csv.DictReader(open(d4 / "labels.csv")):
        rows.append(dict(path=d4 / r["folder"] / r["file"], kind=r["label"], domain="own",
                         group=r["source"], parent=r["parent"], start_s=r["start_s"],
                         anchor=70 if r["label"] != "idle" else -1, src="noah4"))
    # idle from the pauses between swings (person in the scene): only in the older version
    d3 = noah / "_dataset01_010203"
    for r in csv.DictReader(open(d3 / "labels.csv")):
        if r["label"] == "idle":
            rows.append(dict(path=d3 / r["folder"] / r["file"], kind="idle", domain="own",
                             group=r["source"], parent=r["parent"], start_s=r["start_s"], anchor=-1,
                             src="noah3_pause"))
    return rows


def build_rows(build: Path):
    rows = []
    for folder in ("aug_train", "splice_train"):
        for r in csv.DictReader(open(build / "manifest.csv")):
            if r["folder"] != folder:
                continue
            parent = r["parent"].split("#")[0]
            rows.append(dict(path=build / folder / r["file"], kind=r["label"], domain="own_aug",
                             group=parent, parent=parent, start_s="", src=folder,
                             anchor=int(r["release_us"]) // BIN_US if r["release_us"] else -1))
    for folder, dom in (("synth_train", "sim"), ("video_train", "sim")):
        for r in csv.DictReader(open(build / folder / "manifest.csv")):
            g = r.get("person") or r.get("parent") or r["file"]
            rows.append(dict(path=build / folder / r["file"], kind=r["label"], domain=dom,
                             group=f"{folder}:{g}", parent=r.get("parent", ""), start_s="", src=folder,
                             anchor=int(r["release_us"]) // BIN_US if r["release_us"] else -1))
    return rows


def dd_rows(dd: Path):
    rows = []
    for r in csv.DictReader(open(dd / "manifest.csv")):
        kind = {"throw": "throw", "toss": "toss", "arm_no_release": "arm", "idle": "idle"}[r["group"]]
        rows.append(dict(path=dd / r["group"] / r["file"], kind=kind, domain="dd", group=f"dd:{r['person']}",
                         parent=r["parent"], start_s="", src=f"dd:{r['action_name']}", move=r["camera_move"],
                         view=r["perspective"],
                         anchor=int(r["release_us"]) // BIN_US if r["release_us"] else -1))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noah", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--dd", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = own_rows(Path(args.noah)) + build_rows(Path(args.build)) + dd_rows(Path(args.dd))
    codes, offsets, keep = [], [0], []
    for r in rows:
        if not r["path"].exists():
            continue
        c = codes_of(r["path"])
        codes.append(c)
        offsets.append(offsets[-1] + len(c))
        keep.append(r)
    rows = keep
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    col = lambda k: np.array([str(r.get(k, "")) for r in rows])
    np.savez(out, codes=np.concatenate(codes), offsets=np.array(offsets, np.int64),
             y=np.array([int(r["kind"] in POS) for r in rows], np.int64),
             anchor=np.array([int(r["anchor"]) for r in rows], np.int64),
             kind=col("kind"), domain=col("domain"), group=col("group"), parent=col("parent"),
             src=col("src"), start_s=col("start_s"), move=col("move"), view=col("view"), fname=np.array([r["path"].name for r in rows]))
    cells = np.diff(np.array(offsets)) / T
    print(f"{out}: {len(rows)} clips")
    dom = col("domain"); kind = col("kind")
    for d in sorted(set(dom)):
        for k in sorted(set(kind[dom == d])):
            m = (dom == d) & (kind == k)
            print(f"  {d:8s} {k:6s} {m.sum():5d} clips, active cells/frame median {np.median(cells[m]):6.1f}")


if __name__ == "__main__":
    main()
