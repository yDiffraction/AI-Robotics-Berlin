#!/usr/bin/env python3
"""Cut the labelled recordings in dataset/<NN>/samples into 1.2 s clips.

Input: dataset/<NN>/samples/labels.csv, one row per arm swing (written by tools/find_swings.py or
the team's auto labeller, checked with tools/label_tool.py):
  throw    real throw, the ball leaves the hand
  feint    fake throw, the same swing without letting go
  idle     no swing (walking, picking up a ball, arm coming down) -> idle clip
  invalid  unusable, left out               ?  not labelled yet, left out
  end      the usable part of the recording stops here
Labels proposed by a model (training/propose_labels.py) are left out until they are confirmed in
the label tool.

Output (--out, default dataset/_dataset01_010203):
  throw/ feint/ idle/  gesture_<ID>-<label>-<source>-<tag>.bin  (01 throw, 02 feint, 03 idle)
  labels.csv           one row per clip
  preview_<label>.png  contact sheets

Every clip is 1.2 s with timestamps from 0: 0.7 s before to 0.5 s after the decision moment
(event-rate peak of the swing, `release_us` = 700000 for throw and feint). Further idle clips come
from the stretches between the swings (standing, walking, empty scene). Clips cut from the same
swing or the same gap share `parent`; split train/test by `source` or `dataset`, not by clip.

usage: python tools/build_dataset.py [--out dataset/_dataset01_010203] [--verified-only]
"""
import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

import events_tool as ET
from find_swings import decision_key, read_labels, seconds

ROOT = Path(__file__).resolve().parent.parent
WIN, PRE, POST = 1.2, 0.7, 0.5
IDS = {"throw": "01", "feint": "02", "idle": "03"}
COLS = ["file", "folder", "label", "class_id", "dataset", "source", "parent", "start_s", "decision_s",
        "release_us", "events", "verified", "note"]


def cut(ev, t, start):
    """Events of [start, start + WIN) with timestamps relative to `start` (in µs)."""
    lo, hi = np.searchsorted(t, [start, start + WIN])
    sel = ev[lo:hi].copy()
    sel["timestamp"] = np.round((t[lo:hi] - start) * 1e6).astype(np.int64)
    return sel


def idle_starts(busy, t_end):
    """Non-overlapping 1.2 s windows in the gaps between the busy intervals, centred in each gap."""
    out, a = [], 0.0
    for b0, b1 in sorted(busy) + [(t_end, t_end)]:
        n = int((b0 - a + 1e-6) // WIN)
        pad = (b0 - a - n * WIN) / 2
        out += [a + pad + k * WIN for k in range(n)]
        a = max(a, b1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "dataset" / "_dataset01_010203"))
    ap.add_argument("--verified-only", action="store_true", help="leave out rows with verified = 0")
    args = ap.parse_args()
    out = Path(args.out)
    for c in IDS:
        shutil.rmtree(out / c, ignore_errors=True)
        (out / c).mkdir(parents=True)
    clips = []

    for path in sorted((ROOT / "dataset").glob("[0-9]*/samples/labels.csv")):
        ds = path.parent.parent.name
        cols, rows = read_labels(path)
        key = decision_key(cols)
        for f in sorted({r["file"] for r in rows}):
            ev = ET.load(path.parent / f)
            t = seconds(ev)
            src = f"d{ds}_{f.split('_')[2]}"
            mine = [r for r in rows if r["file"] == f]
            end = min([float(r[key]) for r in mine if r["label"] == "end"] + [t[-1]])

            def add(start, label, tag, parent, **kw):
                clip = cut(ev, t, start)
                name = f"gesture_{IDS[label]}-{label}-{src}-{tag}.bin"
                ET.save(out / label / name, clip)
                clips.append({"file": name, "folder": label, "label": label, "class_id": IDS[label],
                              "dataset": ds, "source": f, "parent": f"{f}#{parent}",
                              "start_s": f"{start:.3f}", "events": len(clip), **kw})

            busy = []
            for r in mine:
                d = float(r[key])
                if r["label"] == "end" or d > end:
                    continue
                busy.append((d - PRE + 0.1, d + POST))  # an idle window may share the first 0.1 s
                verified = r.get("verified") or "0"
                note = r.get("note") or r.get("reason") or ""
                proposal = verified == "0" and note.startswith("model:")  # training/propose_labels.py
                if r["label"] not in IDS or proposal or (args.verified_only and verified == "0"):
                    continue
                if d - PRE < 0 or d + POST > t[-1]:
                    print(f"left out, too close to the start or end of the recording: {f} {d:.2f} s")
                    continue
                add(d - PRE, r["label"], f"t{round(d * 100):05d}", f"swing{round(d * 100):05d}",
                    decision_s=f"{d:.3f}", release_us=int(PRE * 1e6) if r["label"] != "idle" else "",
                    verified=verified, note=note)
            for s in idle_starts(busy, end):
                add(s, "idle", f"g{round(s * 100):05d}", f"gap{round(s * 100):05d}", verified="auto",
                    note="between the swings")

    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, restval="")
        w.writeheader()
        w.writerows(sorted(clips, key=lambda r: (r["class_id"], r["file"])))
    sets = sorted({r["dataset"] for r in clips})
    for c in IDS:
        mine = [r for r in clips if r["label"] == c]
        preview(mine, out, c)
        per = ", ".join(f"{d}: {sum(r['dataset'] == d for r in mine)}" for d in sets)
        by = ", ".join(f"{v}: {n}" for v, n in sorted(Counter(r["verified"] for r in mine).items()))
        print(f"{c:5s} {len(mine):3d} clips   dataset {per}   verified {by}")


def preview(rows, out, label, cols=12, px=80, per_sheet=40):
    """One row per clip: 12 frames of 100 ms (ON green, OFF magenta)."""
    from PIL import Image, ImageDraw
    for old in out.glob(f"preview_{label}*.png"):
        old.unlink()
    rows = sorted(rows, key=lambda r: r["file"])
    for k in range(0, len(rows), per_sheet):
        part = rows[k:k + per_sheet]
        sheet = Image.new("RGB", (cols * (px + 1) + 150, len(part) * (px + 1)), (40, 40, 40))
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(part):
            ev = ET.load(out / label / r["file"])
            for c in range(cols):
                acc = ET.accumulate(ev, c * 100_000, (c + 1) * 100_000, 320 // px)
                img = np.zeros((px, px, 3), np.uint8)
                img[..., 1] = np.clip(acc[1] * 120, 0, 255)
                img[..., 0] = img[..., 2] = np.clip(acc[0] * 120, 0, 255)
                sheet.paste(Image.fromarray(img), (c * (px + 1), i * (px + 1)))
            draw.text((cols * (px + 1) + 4, i * (px + 1) + 30), r["file"][8:-4].split("-", 1)[1], fill=(255, 255, 0))
        sheet.save(out / f"preview_{label}_{k // per_sheet + 1}.png")


if __name__ == "__main__":
    main()
