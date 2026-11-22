#!/usr/bin/env python3
"""Convert every event source into one sparse format for the early-throw trainer.

Each clip becomes sorted unique int32 codes of active cells on the board grid
    code = ((frame * 2 + p) * 80 + y) * 80 + x        (80x80, 10 ms frames, binary)
which is the same encoding as training/pretrain/prep_dvsgesture.py.  Output: one .npz per
source with codes / offsets / n_frames and a JSON list of per-clip metadata
(label pos|neg|heldpos, group = recording session / person, rel = release frame).

usage:
  prep_sources.py noah  <dataset dir with labels.csv>   out.npz
  prep_sources.py mb    <motion_buddy_dataset dir>      out.npz [--per 6]
  prep_sources.py dd    <dailydvs dir with manifest.csv> out.npz
  prep_sources.py dvsg  <dvsg_train.npz> <dvsg_test.npz> out.npz
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HW, FRAME_US = 80, 10_000
CELLS = 2 * HW * HW
DT = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])


def read_bin(path):
    return np.frombuffer(open(path, "rb").read()[4:], dtype=DT)


def encode(ev, t0=None, src_hw=320):
    if len(ev) == 0:
        return np.zeros(0, np.int32), 1
    t0 = ev["t"][0] if t0 is None else t0
    f = ((ev["t"] - t0) // FRAME_US).astype(np.int64)
    keep = f >= 0
    x = ev["x"][keep].astype(np.int64) * HW // src_hw
    y = ev["y"][keep].astype(np.int64) * HW // src_hw
    p = (ev["p"][keep] > 0).astype(np.int64)
    c = np.unique(((f[keep] * 2 + p) * HW + y) * HW + x)
    return c.astype(np.int32), int(c.max() // CELLS) + 1 if len(c) else 1


def save(out, codes, nfr, meta):
    off = np.zeros(len(codes) + 1, np.int64)
    off[1:] = np.cumsum([len(c) for c in codes])
    np.savez(out, codes=np.concatenate(codes).astype(np.int32), offsets=off,
             n_frames=np.array(nfr, np.int64), meta=np.array(json.dumps(meta)))
    labs = defaultdict(int)
    for m in meta:
        labs[m["label"]] += 1
    print(out, len(meta), "clips", dict(labs), f"{off[-1] / 1e6:.1f}M codes, {sum(nfr)} frames")


def prep_noah(root, out):
    codes, nfr, meta = [], [], []
    for r in csv.DictReader(open(os.path.join(root, "labels.csv"))):
        ev = read_bin(os.path.join(root, r["folder"], r["file"]))
        # the clip starts at start_s; release_us is relative to the clip start
        c, n = encode(ev, t0=ev["t"][0] if len(ev) else 0)
        n = max(n, 120)
        lab = "neg" if r["label"] == "idle" else "pos"
        rel = int(r["release_us"]) // FRAME_US if r["release_us"] not in ("", "-1") else -1
        grp = "d01" if r["dataset"] == "01" else r["source"]
        codes.append(c); nfr.append(n)
        meta.append(dict(src="noah", label=lab, sub=r["label"], group=f"noah:{grp}", rel=rel,
                         start=float(r["start_s"]), file=r["file"]))
    save(out, codes, nfr, meta)


def prep_mb(root, out, per=6):
    """motion_buddy (Durance, GenX320): sit / lay-down / hello / dance / no-action, folders = sessions.
    Keeps `per` clips per (class-variant, folder) to bound the bank size."""
    by = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, "[0-9][0-9]", "*.bin"))):
        m = re.match(r"gesture_\d+-(.+)-(\d+)\.bin", os.path.basename(f))
        if m:
            by[(m.group(1), os.path.basename(os.path.dirname(f)))].append(f)
    rng = np.random.default_rng(0)
    codes, nfr, meta = [], [], []
    for (cls, folder), fs in sorted(by.items()):
        for f in rng.permutation(fs)[:per]:
            c, n = encode(read_bin(f))
            codes.append(c); nfr.append(n)
            meta.append(dict(src="mb", label="neg", sub=cls, group=f"mb:{folder}", rel=-1,
                             file=os.path.basename(f)))
    save(out, codes, nfr, meta)


def prep_dd(root, out):
    """DailyDVS-200 clips already converted to 320x320 .bin (1.2 s).  throw/toss actions are
    kept as 'heldpos' (an unseen-source positive test set), everything else is negative."""
    codes, nfr, meta = [], [], []
    for r in csv.DictReader(open(os.path.join(root, "manifest.csv"))):
        f = os.path.join(root, r["group"], r["file"])
        if not os.path.exists(f):
            continue
        c, n = encode(read_bin(f))
        lab = "heldpos" if r["group"] in ("throw", "toss") else "neg"
        rel = int(float(r["release_us"])) // FRAME_US if r["release_us"] not in ("", "-1") else -1
        codes.append(c); nfr.append(n)
        meta.append(dict(src="dd", label=lab, sub=r["action_name"], kind=r["group"],
                         group=f"dd:P{r['person']}", rel=rel, file=r["file"]))
    save(out, codes, nfr, meta)


def prep_dvsg(train, test, out):
    names = ["hand_clap", "rh_wave", "lh_wave", "rh_cw", "rh_ccw", "lh_cw", "lh_ccw",
             "arm_roll", "air_drums", "air_guitar", "other"]
    codes, nfr, meta = [], [], []
    for path in (train, test):
        d = np.load(path)
        for i in range(len(d["labels"])):
            c = d["codes"][d["offsets"][i]:d["offsets"][i + 1]].astype(np.int64)
            codes.append(c.astype(np.int32)); nfr.append(int(d["n_frames"][i]))
            meta.append(dict(src="dvsg", label="neg", sub=names[int(d["labels"][i])],
                             group=f"dvsg:U{int(d['users'][i])}", rel=-1))
    save(out, codes, nfr, meta)


if __name__ == "__main__":
    kind, args = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
    if kind == "noah":
        prep_noah(*args)
    elif kind == "mb":
        per = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 6
        prep_mb(args[0], args[1], per)
    elif kind == "dd":
        prep_dd(*args)
    elif kind == "dvsg":
        prep_dvsg(*args)
