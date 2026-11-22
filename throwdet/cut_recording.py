#!/usr/bin/env python3
"""Cut new GenX320 recordings (e.g. made at the demo site) into 1.2 s training clips for adapt.

Two kinds of input, combinable:
  --idle REC.bin ...    recordings WITHOUT any throw motion (people standing, walking, the scene
                        at the demo site). Cut into contiguous 1.2 s idle clips. No labels needed.
  --hints hints.csv     hints from tools/record_tool.py (noah branch): file, label (throw/feint),
                        hint_s. The swing is the event-rate peak in [hint - 2.0 s, hint + 0.3 s]
                        (the key is pressed 0.5-1.5 s after the swing); clip = peak - 0.7 s ..
                        peak + 0.5 s. Time away from all swings (> 2.5 s) becomes idle clips
                        with --pauses.

Output: <out>/{throw,feint,idle}/gesture_*.bin and <out>/labels.csv in the format of the noah
branch datasets (file, folder, label, source, parent, start_s, decision_s, release_us), which
throwdet/prep.py reads with --own-new.

  python throwdet/cut_recording.py --idle site_idle.bin --hints dataset/05/samples/hints.csv \
      --rec-dir dataset/05/samples --pauses --out ~/throwdet/site
"""
import argparse
import csv
import struct
from pathlib import Path

import numpy as np

DT = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])
CLASS_ID = {"throw": 1, "feint": 2, "idle": 3}


def load(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("!I", raw[:4])[0]
    ev = np.frombuffer(raw, DT, offset=4, count=n // DT.itemsize)
    t = ev["t"].astype(np.int64) - int(ev["t"][0])
    return ev, t


def save(path, ev, t0):
    out = ev.copy()
    out["t"] = ev["t"].astype(np.int64) - t0
    body = out.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("!I", len(body)) + body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idle", nargs="*", default=[], help="recordings without any throw motion")
    ap.add_argument("--hints", default=None)
    ap.add_argument("--rec-dir", default=None, help="folder of the hinted recordings (default: next to hints.csv)")
    ap.add_argument("--pauses", action="store_true", help="also cut idle clips from the pauses of hinted recordings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    rows = []

    def idle_clips(path, ev, t, avoid=()):
        tag = Path(path).stem.replace("sample_", "")
        for s in range(0, int(t[-1]) - 1_200_000, 1_200_000):
            if any(abs(s + 600_000 - a) < 2_500_000 for a in avoid):
                continue
            m = (t >= s) & (t < s + 1_200_000)
            if m.sum() < 50:
                continue
            name = f"gesture_03-idle-new_{tag}-g{s // 10_000:05d}.bin"
            save(out / "idle" / name, ev[m], int(ev["t"][0]) + s)
            rows.append({"file": name, "folder": "idle", "label": "idle", "source": Path(path).name,
                         "parent": f"{Path(path).name}#gap{s // 10_000:05d}", "start_s": f"{s / 1e6:.3f}",
                         "decision_s": "", "release_us": ""})

    for path in args.idle:
        ev, t = load(path)
        idle_clips(path, ev, t)

    if args.hints:
        rec_dir = Path(args.rec_dir or Path(args.hints).parent)
        hints = [h for h in csv.DictReader(open(args.hints)) if h["label"] in ("throw", "feint")]
        for fname in sorted({h["file"] for h in hints}):
            ev, t = load(rec_dir / fname)
            edges = np.arange(0, t[-1] + 20_000, 20_000)
            rate = np.convolve(np.histogram(t, edges)[0], np.ones(3) / 3, mode="same")
            peaks = []
            for h in [h for h in hints if h["file"] == fname]:
                hs = int(float(h["hint_s"]) * 1e6)
                lo, hi = np.searchsorted(edges, [hs - 2_000_000, hs + 300_000])
                if hi <= lo:
                    continue
                pk = int(edges[lo + int(np.argmax(rate[lo:hi]))] + 10_000)
                if pk < 700_000 or pk + 500_000 > t[-1] or any(abs(pk - p) < 400_000 for p in peaks):
                    continue
                peaks.append(pk)
                m = (t >= pk - 700_000) & (t < pk + 500_000)
                tag = fname.replace("sample_", "").replace(".bin", "")
                name = f"gesture_{CLASS_ID[h['label']]:02d}-{h['label']}-new_{tag}-t{pk // 10_000:05d}.bin"
                save(out / h["label"] / name, ev[m], int(ev["t"][0]) + pk - 700_000)
                rows.append({"file": name, "folder": h["label"], "label": h["label"], "source": fname,
                             "parent": f"{fname}#swing{pk // 10_000:05d}", "start_s": f"{(pk - 700_000) / 1e6:.3f}",
                             "decision_s": f"{pk / 1e6:.3f}", "release_us": 700000})
            if args.pauses:
                idle_clips(rec_dir / fname, ev, t, avoid=peaks + [int(float(h["hint_s"]) * 1e6) for h in hints
                                                                   if h["file"] == fname])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "folder", "label", "source", "parent", "start_s", "decision_s", "release_us"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"{out}: {len(rows)} clips {dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
