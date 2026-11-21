#!/usr/bin/env python3
"""Build the throw / feint / idle training set from our recordings.

Pipeline
  1. Auto-label every raw recording from its motion profile (see `auto_label`) and write
     <raw>/labels.csv. Rows with status=confirmed (set by a human) are kept as they are.
  2. Cut each valid recording to a fixed window around the decision moment
     (`pre` s before to `post` s after the event-rate peak = arm swing / release).
     Quiet stretches before the wind-up become `idle` windows.
  3. Split by source recording (train / val / test); variants always stay in the split
     of the recording they come from. val and test hold only unmodified real windows.
  4. Grow the training split:
       aug     label-preserving event augmentation of real windows
               (mirror, shift, scale, rotation, speed, dropout, noise, time jitter, distractors)
       cutrev  synthetic feints from real throws: stop before release, replay the wind-up backwards
       synth   procedural renderer + event simulator (tools/synth_throw.py)
       durance other human motion from the Durance gesture dataset as extra `idle` (local only)
  5. Write manifest.csv (one row per file) and MP4 review videos / reels.

Output layout (each folder is a "session" for training/train_snn.py):
  <out>/real_train  real_val  real_test  aug_train  cutrev_train  synth_train  durance_train
  <out>/manifest.csv  <out>/videos/*.mp4

  uv run tools/build_dataset.py --raw dataset/01/samples --out dataset/build
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import event_aug as A          # noqa: E402
import events_tool as ET       # noqa: E402
import events_video as V       # noqa: E402

CLASS_ID = {"throw": 1, "feint": 2, "idle": 3}


# --- 1. analysis + auto labels ----------------------------------------------------

def motion_profile(ev, bin_us=20_000):
    ts = ev["timestamp"].astype(np.int64)
    t, on, off = ET.rate_curve(ev, bin_us)
    rate = (on + off).astype(float)
    k = int(np.argmax(rate))
    peak = rate[k]
    l = k
    while l > 0 and rate[l] > peak / 2:
        l -= 1
    h = k
    while h < len(rate) - 1 and rate[h] > peak / 2:
        h += 1
    pre = rate[max(0, k - 25):max(1, k - 5)].mean() + 1
    post = rate[k + 5:k + 25].mean() + 1 if k + 5 < len(rate) else 0.0
    # wind-up start: last time before the peak where activity was below 15 % of the peak
    quiet = np.nonzero(rate[:k] < 0.15 * peak)[0]
    windup = quiet[-1] if len(quiet) else 0
    return {
        "start_us": int(ts[0]), "end_us": int(ts[-1]),
        "peak_us": int(ts[0] + k * bin_us + bin_us // 2),
        "windup_us": int(ts[0] + windup * bin_us),
        "peak_rate": float(peak), "width_ms": (h - l) * bin_us // 1000,
        "post_pre": round(float(post / pre), 3),
        "peak_at_end": k >= len(rate) - 4,
    }


def auto_label(m):
    """Throw: one sharp swing, then the scene goes quiet. Feint: activity continues after the
    peak because the arm stops and comes back. Thresholds were set on dataset/01 (2026-11-21)."""
    if m["peak_at_end"]:
        return "invalid", 1.0, "peak at clip end (cut off)"
    if m["post_pre"] < 0.55 and m["width_ms"] <= 200:
        conf = 0.9 if m["post_pre"] < 0.4 else 0.7
        return "throw", conf, "sharp swing, quiet afterwards"
    if m["post_pre"] > 1.0:
        return "feint", 0.85, "activity continues after the swing"
    if m["post_pre"] > 0.6:
        return "feint", 0.6, "moderate activity after the swing"
    return "throw", 0.5, "ambiguous"


def file_time_s(name):
    """sample_YYYYMMDD_HHMMSS_micro.bin -> seconds of the day."""
    parts = Path(name).stem.split("_")
    hhmmss = parts[2]
    return int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6]) + int(parts[3]) / 1e6


def smooth_by_recording_order(rows, max_gap_s=90.0):
    """Takes were recorded in blocks (all throws, then all feints). An uncertain auto label
    takes the label of its neighbours when both neighbours agree with confidence >= 0.7."""
    valid = [r for r in rows if r["label"] in ("throw", "feint")]
    for i, r in enumerate(valid):
        if r["status"] != "auto" or float(r["confidence"]) > 0.6 or i == 0 or i == len(valid) - 1:
            continue
        a, b = valid[i - 1], valid[i + 1]
        t = file_time_s(r["file"])
        close = t - file_time_s(a["file"]) <= max_gap_s and file_time_s(b["file"]) - t <= max_gap_s
        if close and a["label"] == b["label"] and min(float(a["confidence"]), float(b["confidence"])) >= 0.7:
            r["reason"] = f"recording block ({r['label']} by motion, {a['label']} by neighbours)"
            r["label"], r["confidence"] = a["label"], 0.7


def label_raw(raw_dir):
    path = Path(raw_dir) / "labels.csv"
    old = {}
    if path.exists():
        with path.open() as f:
            old = {r["file"]: r for r in csv.DictReader(f)}
    rows = []
    for f in sorted(Path(raw_dir).glob("*.bin")):
        ev = ET.load(f)
        m = motion_profile(ev)
        if f.name in old and old[f.name].get("status") == "confirmed":
            label, conf, why, status = old[f.name]["label"], 1.0, old[f.name].get("reason", "confirmed"), "confirmed"
        else:
            label, conf, why = auto_label(m)
            status = "auto"
        rows.append({"file": f.name, "label": label, "confidence": conf, "status": status, "reason": why,
                     "peak_s": round((m["peak_us"] - m["start_us"]) / 1e6, 3),
                     "windup_s": round((m["windup_us"] - m["start_us"]) / 1e6, 3),
                     "duration_s": round((m["end_us"] - m["start_us"]) / 1e6, 3),
                     "peak_rate": int(m["peak_rate"]), "width_ms": m["width_ms"], "post_pre": m["post_pre"]})
    smooth_by_recording_order(rows)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# --- helpers ----------------------------------------------------------------------

class Out:
    def __init__(self, root):
        self.root = Path(root)
        self.rows = []

    def save(self, folder, label, source, parent, ev, release_us, tag, **extra):
        d = self.root / folder
        d.mkdir(parents=True, exist_ok=True)
        name = f"gesture_{CLASS_ID[label]:02d}-{label}-{source}-{tag}.bin"
        ET.save(d / name, ev)
        self.rows.append({"file": name, "folder": folder, "split": folder.split("_")[-1], "label": label,
                          "source": source, "parent": parent, "release_us": "" if release_us is None else int(release_us),
                          "events": len(ev), **extra})
        return d / name

    def manifest(self):
        keys = sorted({k for r in self.rows for k in r}, key=lambda k: (k not in ("file", "folder", "split", "label", "source", "parent", "release_us", "events"), k))
        with (self.root / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def assign_splits(items):
    """Per class, in recording order: every 5th (offset 2) -> test, every 5th (offset 4) -> val."""
    split = {}
    for label in ("throw", "feint"):
        group = [r for r in items if r["label"] == label]
        for i, r in enumerate(group):
            split[r["file"]] = "test" if i % 5 == 2 else "val" if i % 5 == 4 else "train"
    return split


def augment(ev, rng, duration_us, release_us, distractors):
    out = A.geometric(ev, rng)
    factor = rng.uniform(0.8, 1.25)
    out = A.time_stretch(out, factor, release_us)
    out = A.dropout(out, rng)
    if distractors and rng.random() < 0.3:
        out = A.overlay(out, distractors[rng.integers(len(distractors))], rng)
    out = A.add_noise(out, rng, duration_us)
    return out, factor


# --- main -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="dataset/01/samples")
    p.add_argument("--out", default="dataset/build")
    p.add_argument("--pre", type=float, default=0.7, help="seconds kept before the peak")
    p.add_argument("--post", type=float, default=0.3, help="seconds kept after the peak")
    p.add_argument("--aug-per-clip", type=int, default=25)
    p.add_argument("--cutrev-per-throw", type=int, default=10)
    p.add_argument("--synth-per-class", type=int, default=120)
    p.add_argument("--durance", default="data/motion_buddy_dataset", help="Durance dataset root (optional)")
    p.add_argument("--durance-count", type=int, default=150)
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Out(args.out)
    win = int((args.pre + args.post) * 1e6)
    pre = int(args.pre * 1e6)

    rows = label_raw(args.raw)
    print("labels:", {k: sum(r["label"] == k for r in rows) for k in ("throw", "feint", "invalid")})
    usable = [r for r in rows if r["label"] in ("throw", "feint") and float(r["confidence"]) >= args.min_confidence]
    split = assign_splits(usable)

    # distractors: other human motion (Durance gesture dataset, sessions 06+), local only
    distractor_files = sorted(glob.glob(f"{args.durance}/*/gesture_*.bin"))
    distractor_files = [f for f in distractor_files if Path(f).parent.name >= "06"]
    rng.shuffle(distractor_files)
    distractors = [A.as_native(ET.load(f)) for f in distractor_files[:30]]
    for d in distractors:
        d["timestamp"] -= d["timestamp"][0]

    real_windows = {}
    for r in usable:
        ev = A.as_native(ET.load(Path(args.raw) / r["file"]))
        m = motion_profile(ev)
        s = split[r["file"]]
        t0 = max(m["start_us"], m["peak_us"] - pre)
        w = A.window(ev, t0, t0 + win)
        rel = m["peak_us"] - t0
        real_windows[r["file"]] = (w, rel, s, r["label"])
        stem = Path(r["file"]).stem.replace("sample_", "")
        out.save(f"real_{s}", r["label"], "real", r["file"], w, rel, stem, confidence=r["confidence"])
        # idle: quiet stretch before the wind-up (person standing, holding the ball)
        if m["windup_us"] - m["start_us"] >= 0.6e6:
            iw = A.window(ev, m["start_us"], min(m["windup_us"], m["start_us"] + win))
            out.save(f"real_{s}", "idle", "real", r["file"], iw, None, stem + "-pre")
        # idle: after a throw the thrower stands still again
        if r["label"] == "throw" and m["end_us"] - (m["peak_us"] + 400_000) >= 0.6e6:
            iw = A.window(ev, m["peak_us"] + 400_000, min(m["end_us"], m["peak_us"] + 400_000 + win))
            out.save(f"real_{s}", "idle", "real", r["file"], iw, None, stem + "-post")

    train = {f: v for f, v in real_windows.items() if v[2] == "train"}
    print(f"real windows: {len(real_windows)} ({sum(v[2] == 'train' for v in real_windows.values())} train)")

    # augmentation of real training windows (throw, feint and their idle parts)
    for f, (w, rel, _, label) in train.items():
        stem = Path(f).stem.replace("sample_", "")
        for k in range(args.aug_per_clip):
            jitter = int(rng.uniform(-0.1, 0.1) * 1e6)
            shifted = w.copy()
            shifted["timestamp"] -= jitter
            shifted = shifted[(shifted["timestamp"] >= 0) & (shifted["timestamp"] < win)]
            ev, factor = augment(shifted, rng, win, rel - jitter, distractors)
            ev = ev[(ev["timestamp"] >= 0) & (ev["timestamp"] < win)]
            out.save("aug_train", label, "aug", f, ev, (rel - jitter), f"{stem}-a{k:03d}", stretch=round(factor, 3))

    # synthetic feints from real training throws
    for f, (w, rel, _, label) in train.items():
        if label != "throw":
            continue
        stem = Path(f).stem.replace("sample_", "")
        for k in range(args.cutrev_per_throw):
            ev, stop = A.feint_from_throw(w, rel, rng)
            if rng.random() < 0.7:
                ev, _ = augment(ev, rng, win, stop, distractors)
            ev = ev[(ev["timestamp"] >= 0) & (ev["timestamp"] < win)]
            out.save("cutrev_train", "feint", "cutrev", f, ev, stop, f"{stem}-c{k:03d}")

    # Durance gestures as idle (non-throw human motion)
    for k, f in enumerate(distractor_files[30:30 + args.durance_count]):
        ev = A.as_native(ET.load(f))
        ev["timestamp"] -= ev["timestamp"][0]
        ev = A.geometric(ev, rng, scale=(0.7, 1.1))
        out.save("durance_train", "idle", "durance", Path(f).name, ev, None, f"{k:04d}")

    # procedural synthetic clips
    if args.synth_per_class:
        import synth_throw as S
        for label in ("throw", "feint", "idle"):
            for k in range(args.synth_per_class):
                view = "profile" if k % 3 else "frontal"       # 2/3 match our camera setup
                ev, _, meta = S.make_clip(rng, label, view, 1.3, 500)
                ev = A.as_native(ev)
                if label != "idle":
                    t0 = max(0, meta["release_us"] - pre)
                    ev = A.window(ev, t0, t0 + win)
                    rel = meta["release_us"] - t0
                else:
                    ev = A.window(ev, 0, win)
                    rel = None
                out.save("synth_train", label, "synth", "", ev, rel, f"{view}-{k:04d}",
                         view=view, idle_kind=meta["idle_kind"], feint_fraction=meta["feint_fraction"])
            print(f"synth {label}: {args.synth_per_class}")

    out.manifest()
    counts = {}
    for r in out.rows:
        counts.setdefault(r["folder"], {}).setdefault(r["label"], 0)
        counts[r["folder"]][r["label"]] += 1
    for folder, c in sorted(counts.items()):
        print(f"{folder:15s} {c}")

    if not args.no_video:
        vids = Path(args.out) / "videos"
        rel = {r["file"]: r["release_us"] for r in out.rows}
        # review: every raw recording, full length, with its (auto) label
        lab = {r["file"]: r for r in rows}
        files = sorted(Path(args.raw).glob("*.bin"))
        tmp = Path(args.out) / "videos" / "_review"
        tmp.mkdir(parents=True, exist_ok=True)
        for f in files:
            r = lab[f.name]
            V.write_clip_video(ET.load(f), tmp / (f.stem + ".mp4"), label=r["label"],
                               release_us=int(float(r["peak_s"]) * 1e6),
                               title=f"{f.stem[-13:]} {r['status']} {float(r['confidence']):.1f}", speed=0.5)
        print("review videos:", tmp)
        for folder in sorted(counts):
            sel = sorted((Path(args.out) / folder).glob("*.bin"))
            pick = [sel[i] for i in np.linspace(0, len(sel) - 1, min(len(sel), 24)).astype(int)]
            V.write_reel([str(x) for x in pick], vids / f"reel_{folder}.mp4", (4, 3), args.pre + args.post,
                         releases=rel)
            print("reel:", vids / f"reel_{folder}.mp4")


if __name__ == "__main__":
    main()
