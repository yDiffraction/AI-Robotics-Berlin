#!/usr/bin/env python3
"""Raw GenX320 recordings -> per-recording event-count tensors for robust_cv.py.

Each recording becomes counts [F, 2, 80, 80] uint8 (10 ms frames, 4x4 blocks = despot
downsampling 4, saturating at 255) over its full length, so windows can be cut on the fly
with real context instead of zero padding. Labels and the swing peak come from labels.csv
(tools/build_dataset.py); rows labelled `invalid` are skipped.

Optionally also packs an offline-generated folder (e.g. dataset/build/aug_train) as 1.2 s
windows with release at 700 ms, keyed by parent recording so CV can exclude derivatives.

usage: python -I prep_raw.py <samples_dir> <out.pt> [--extra dataset/build aug_train,splice_train]
       python -I prep_raw.py dataset/_dataset01_010203 bank_clips.pt --clips
"""
import argparse
import csv
import struct
from pathlib import Path

import numpy as np
import torch

DT = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])
SENSOR, HW, BIN_US = 320, 80, 10_000
LABELS = {"throw": 0, "feint": 1, "idle": 2}


def counts(path, t0_us=0, frames=None):
    raw = Path(path).read_bytes()
    n = struct.unpack("!I", raw[:4])[0]
    assert n == len(raw) - 4 and n % DT.itemsize == 0, path
    ev = np.frombuffer(raw, DT, offset=4)
    f = (ev["t"].astype(np.int64) - int(ev["t"][0]) - t0_us) // BIN_US
    F_ = int(f.max()) + 1 if frames is None else frames
    keep = (f >= 0) & (f < F_) & (ev["x"] < SENSOR) & (ev["y"] < SENSOR) & (ev["p"] < 2)
    b = SENSOR // HW
    idx = ((f[keep] * 2 + ev["p"][keep]) * HW + ev["y"][keep] // b) * HW + ev["x"][keep] // b
    c = np.bincount(idx, minlength=F_ * 2 * HW * HW).clip(0, 255).astype(np.uint8)
    return torch.from_numpy(c.reshape(F_, 2, HW, HW))


def prep_clips(root):
    """Labelled 1.2 s clip dataset (e.g. dataset/_dataset01_010203): release at 700 ms."""
    recs = []
    for r in csv.DictReader(open(Path(root) / "labels.csv")):
        if r["label"] not in LABELS:
            continue
        lab = LABELS[r["label"]]
        group = "d01" if r["dataset"] == "01" else r["source"]      # dataset 01 = one recording session
        # binary cells: idle clips may be pre-thinned (one event per 10 ms / 4x4 block), so raw counts would
        # differ by class and event thinning would hit idle harder -> store what the model sees
        c = counts(Path(root) / r["folder"] / r["file"], frames=120).clamp(max=1)
        recs.append({"name": r["file"], "label": lab, "counts": c, "start": float(r["start_s"] or 0),
                     "peak": int(r["release_us"]) // BIN_US if r["release_us"] else -1, "windup": -1,
                     "conf": 1.0, "clip": True, "group": group, "verified": r["verified"]})
    print(f"{len(recs)} clips, classes {np.bincount([r['label'] for r in recs]).tolist()}, "
          f"groups {sorted({r['group'] for r in recs})}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("out")
    ap.add_argument("--extra", nargs=2, metavar=("BUILD", "FOLDERS"))
    ap.add_argument("--clips", action="store_true", help="samples is a labelled clip dataset")
    a = ap.parse_args()
    if a.clips:
        torch.save({"recs": prep_clips(a.samples), "extra": []}, a.out)
        return
    recs = []
    for r in csv.DictReader(open(Path(a.samples) / "labels.csv")):
        if r["label"] not in LABELS:
            continue
        c = counts(Path(a.samples) / r["file"])
        recs.append({"name": r["file"], "label": LABELS[r["label"]], "counts": c,
                     "peak": int(round(float(r["peak_s"]) * 100)), "windup": int(round(float(r["windup_s"]) * 100)),
                     "conf": float(r["confidence"])})
        print(f"{r['file']} {r['label']:5s} frames {len(c)} peak {recs[-1]['peak']} windup {recs[-1]['windup']}")
    extra = []
    if a.extra:
        build, folders = Path(a.extra[0]), a.extra[1].split(",")
        rows = list(csv.DictReader(open(build / "manifest.csv")))
        for r in rows:
            if r["folder"] in folders and r["label"] in LABELS:
                rel = int(r["release_us"]) // BIN_US if r["release_us"] else -1
                extra.append({"folder": r["folder"], "parent": r["parent"], "donor": r.get("donor", ""),
                              "label": LABELS[r["label"]], "release": rel,
                              "counts": counts(build / r["folder"] / r["file"], frames=120)})
        print(f"extra windows: {len(extra)}")
    torch.save({"recs": recs, "extra": extra}, a.out)


if __name__ == "__main__":
    main()
