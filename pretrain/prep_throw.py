"""Convert throw/feint/idle windows (dataset/build, GenX320 .bin) into the board's input format.

Each window (1.2 s, timestamps from 0) becomes binary [T, 2, 80, 80] frames of 10 ms, stored as
sorted active-cell codes like the DVS Gesture npz (code = ((frame*2+p)*80+y)*80+x).
320x320 -> 80x80 is a 4x4 block (despot downsampling 4); a cell is 1 if any event fell in it.

usage: python -I prep_throw.py <build_dir> <out_dir>
  writes <out_dir>/throw_real.npz (real_train+real_val+real_test) and throw_synth.npz (synth_train)
"""
from __future__ import annotations

import csv
import struct
import sys
from pathlib import Path

import numpy as np

DT = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])
SENSOR, HW, BIN_US, T = 320, 80, 10_000, 120
LABELS = {"throw": 0, "feint": 1, "idle": 2}
REAL = ("real_train", "real_val", "real_test")


def codes_of(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = struct.unpack("!I", raw[:4])[0]
    assert n == len(raw) - 4 and n % DT.itemsize == 0, path
    ev = np.frombuffer(raw, DT, offset=4)
    f = ev["t"].astype(np.int64) // BIN_US
    keep = (f >= 0) & (f < T) & (ev["x"] < SENSOR) & (ev["y"] < SENSOR) & (ev["p"] < 2)
    b = SENSOR // HW
    c = ((f[keep] * 2 + ev["p"][keep]) * HW + ev["y"][keep] // b) * HW + ev["x"][keep] // b
    return np.unique(c.astype(np.uint32))


def write(rows, build: Path, out: Path):
    codes, offsets = [], [0]
    for r in rows:
        c = codes_of(build / r["folder"] / r["file"])
        codes.append(c)
        offsets.append(offsets[-1] + len(c))
    rel = [int(r["release_us"]) // BIN_US if r.get("release_us") not in ("", None) else -1 for r in rows]
    np.savez(out, codes=np.concatenate(codes), offsets=np.array(offsets, np.int64),
             labels=np.array([LABELS[r["label"]] for r in rows], np.int64),
             release_frame=np.array(rel, np.int64),
             parent=np.array([r.get("parent") or r["file"] for r in rows]),
             folder=np.array([r["folder"] for r in rows]),
             view=np.array([r.get("view", "") for r in rows]),
             fname=np.array([r["file"] for r in rows]),
             n_frames=np.full(len(rows), T, np.int64), users=np.zeros(len(rows), np.int64))
    lab = np.bincount([LABELS[r["label"]] for r in rows], minlength=3)
    print(f"{out}: {len(rows)} windows (throw/feint/idle {lab.tolist()}), "
          f"{offsets[-1] / len(rows) / T:.0f} active cells per frame", flush=True)


def main():
    build, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    real = [r for r in csv.DictReader(open(build / "manifest.csv")) if r["folder"] in REAL]
    real = [r for r in real if r["label"] in LABELS]
    write(real, build, out / "throw_real.npz")
    synth = [dict(r, folder="synth_train") for r in csv.DictReader(open(build / "synth_train" / "manifest.csv"))]
    synth = [r for r in synth if r["label"] in LABELS]
    write(synth, build, out / "throw_synth.npz")


if __name__ == "__main__":
    main()
