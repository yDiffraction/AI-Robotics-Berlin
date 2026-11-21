#!/usr/bin/env python3
"""Turn ordinary (ideally 240 fps slow-motion) phone videos into GenX320-style event clips.

Why: the ball is hard to see for the event camera and real camera time is scarce, but
anyone can film many people throwing / feinting with a phone. In the RGB video the
ball is visible, so labels and release times are reliable. Real people, clothes and
backgrounds give the realism the procedural renderer lacks.

Input: a CSV (one row per take), columns
  video       path to the video file
  label       throw | feint | idle
  release_s   throw: time the ball leaves the hand; feint: time the arm stops
              (seconds in the video's own timeline; empty for idle or to auto-detect)
  start_s, end_s   optional trim
  true_fps    optional: real capture rate if the file plays slowed down
              (e.g. a slow-motion export at 30 fps that was shot at 240 fps -> 240)
  person, session, notes   optional metadata (used for grouped splits)
  crop        optional "x,y,size" square in source pixels (default: automatic, around the motion)

Per take it writes `--variants` clips: each with freshly sampled sensor parameters
(tools/event_sim.SensorParams), crop jitter (scale/shift), optional mirror, brightness
and gamma changes, and release-time jitter. Output: .bin clips named
gesture_<ID>-<label>-video-<tag>.bin, manifest.csv and MP4 previews (RGB | events).

  uv run tools/video_to_events.py --takes recording/videos/takes.csv --out dataset/video --variants 8
"""

import argparse
import csv
import re
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import event_sim  # noqa: E402

CLASS_ID = {"throw": 1, "feint": 2, "idle": 3}
SENSOR = event_sim.SENSOR
WORK = 400          # decoded working resolution of the (enlarged) crop region


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return s["width"], s["height"], num / den


def decode(path, crop, size, start=None, end=None):
    """Decode a square crop (x, y, side in source pixels) as grayscale frames [N, size, size]."""
    x, y, side = crop
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if end:
        cmd += ["-to", str(end)]
    cmd += ["-i", str(path), "-vf", f"crop={side}:{side}:{x}:{y},scale={size}:{size},format=gray",
            "-vsync", "0", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size).astype(np.float32) / 255.0


def auto_crop(path, w, h, start=None, end=None, margin=1.5):
    """Square around the region with the most frame-to-frame change (the moving person)."""
    side = min(w, h)
    small = decode(path, ((w - side) // 2, (h - side) // 2, side), 96, start, end)
    if len(small) < 3:
        return (w - side) // 2, (h - side) // 2, side
    energy = np.abs(np.diff(small, axis=0)).sum(0)
    ys, xs = np.nonzero(energy > np.percentile(energy, 97))
    scale = side / 96
    cx, cy = (xs.mean() + 0.5) * scale + (w - side) // 2, (ys.mean() + 0.5) * scale + (h - side) // 2
    extent = max(np.ptp(xs), np.ptp(ys), 20) * scale * margin
    s = int(np.clip(extent, side * 0.4, side))
    x = int(np.clip(cx - s / 2, 0, w - s))
    y = int(np.clip(cy - s / 2, 0, h - s))
    return x, y, s


def motion_peak_s(frames, fps):
    """Fallback release proxy: frame with the largest total change (arm swing peak)."""
    e = np.abs(np.diff(frames, axis=0)).reshape(len(frames) - 1, -1).sum(1)
    e = np.convolve(e, np.ones(5) / 5, mode="same")
    return (int(np.argmax(e)) + 1) / fps


def variant_frames(frames, rng, flip):
    """Crop jitter (scale 0.8-1.0 of the working crop, shift), mirror, brightness/gamma -> SENSOR px."""
    n = frames.shape[1]
    s = int(n * rng.uniform(0.8, 1.0))
    x0 = int(rng.uniform(0, n - s))
    y0 = int(rng.uniform(0, n - s))
    gain = rng.uniform(0.6, 1.3)
    gamma = rng.uniform(0.8, 1.25)
    out = np.empty((len(frames), SENSOR, SENSOR), np.float32)
    for i, f in enumerate(frames):
        img = Image.fromarray((f[y0:y0 + s, x0:x0 + s] * 255).astype(np.uint8)).resize((SENSOR, SENSOR), Image.BILINEAR)
        a = np.asarray(img, np.float32) / 255.0
        out[i] = np.clip(gain * a ** gamma, 0, 1)
    if flip:
        out = out[:, :, ::-1]
    return out, {"crop_scale": round(s / n, 3), "crop_x": x0, "crop_y": y0, "gain": round(gain, 3),
                 "gamma": round(gamma, 3), "flip": bool(flip)}


def save_bin(path, ev):
    import struct
    body = ev.tobytes()
    Path(path).write_bytes(struct.pack("!I", len(body)) + body)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--takes", required=True, help="CSV with one row per take (see module doc)")
    p.add_argument("--out", required=True)
    p.add_argument("--variants", type=int, default=8)
    p.add_argument("--pre", type=float, default=0.7, help="seconds before the release kept in a clip")
    p.add_argument("--post", type=float, default=0.5, help="seconds after the release kept in a clip")
    p.add_argument("--jitter", type=float, default=0.1, help="random shift of the window (s)")
    p.add_argument("--idle-len", type=float, default=1.2)
    p.add_argument("--max-fps", type=float, default=1000.0, help="frame rate cap used for simulation")
    p.add_argument("--previews", type=int, default=1, help="MP4 previews per take (first N variants)")
    p.add_argument("--rate-k", default="60,300",
                   help="event-rate range of the real GenX320 clips (k ev/s); denser clips are thinned "
                        "to a random target in this range")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    rate_k = [float(v) for v in args.rate_k.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    takes = list(csv.DictReader(open(args.takes)))
    base = Path(args.takes).parent
    rows = []
    for ti, take in enumerate(takes):
        video = Path(take["video"])
        video = video if video.is_absolute() else base / video
        label = take["label"].strip()
        w, h, file_fps = probe(video)
        true_fps = float(take.get("true_fps") or file_fps)
        start = float(take["start_s"]) if take.get("start_s") else None
        end = float(take["end_s"]) if take.get("end_s") else None
        if take.get("crop"):
            cx, cy, cs = map(int, take["crop"].split(","))
        else:
            cx, cy, cs = auto_crop(video, w, h, start, end)
        # enlarge the crop a little so variants can zoom/shift inside it
        grow = int(cs * 0.15)
        side = min(cs + 2 * grow, w, h)
        cx = int(np.clip(cx - grow, 0, w - side)); cy = int(np.clip(cy - grow, 0, h - side))
        frames = decode(video, (cx, cy, side), WORK, start, end)
        step = max(1, int(round(true_fps / args.max_fps)))
        frames = frames[::step]
        fps = true_fps / step
        times = np.arange(len(frames)) * 1e6 / fps
        if take.get("release_s"):
            release = float(take["release_s"]) - (start or 0.0)
            release_src = "annotated"
        elif label != "idle":
            release = motion_peak_s(frames, fps)
            release_src = "motion_peak"
        else:
            release, release_src = None, ""
        safe = lambda v, d: re.sub(r"[^A-Za-z0-9_]+", "_", v or d).strip("_")
        stem = f"{safe(take.get('session'), 'v')}-{safe(take.get('person'), 'p')}-{ti:03d}"
        for k in range(args.variants):
            if release is not None:
                j = rng.uniform(-args.jitter, args.jitter)
                t0 = max(0.0, release - args.pre + j)
                t1 = min(times[-1] / 1e6, release + args.post + j)
                rel_us = int((release - t0) * 1e6)
            else:
                length = min(args.idle_len, times[-1] / 1e6)
                t0 = rng.uniform(0, max(0.0, times[-1] / 1e6 - length))
                t1, rel_us = t0 + length, None
            sel = (times >= t0 * 1e6) & (times <= t1 * 1e6)
            if sel.sum() < 4:
                continue
            vf, aug = variant_frames(frames[sel], rng, flip=rng.random() < 0.5)
            params = event_sim.SensorParams.sample(rng)
            ev = event_sim.simulate(vf, times[sel] - times[sel][0], params, rng)
            # phone videos (handheld, large moving area) give far more events than the real camera
            target = rng.uniform(*rate_k) * 1e3 * (t1 - t0)
            if len(ev) > target:
                ev = ev[np.sort(rng.choice(len(ev), int(target), replace=False))]
            name = f"gesture_{CLASS_ID[label]:02d}-{label}-video-{stem}-v{k:02d}.bin"
            save_bin(out / name, ev)
            rows.append({"file": name, "label": label, "source": "video", "parent": str(take["video"]),
                         "person": take.get("person", ""), "session": take.get("session", ""),
                         "release_us": "" if rel_us is None else rel_us, "release_source": release_src,
                         "events": len(ev), "fps": round(fps, 1), "duration_us": int((t1 - t0) * 1e6),
                         "params": json.dumps({**params.as_dict(), **aug})})
            if k < args.previews:
                import events_video
                events_video.write_clip_video(ev, out / "videos" / name.replace(".bin", ".mp4"), label=label,
                                              release_us=rel_us, source_frames=vf, source_fps=fps)
            print(f"{name}  {len(ev)} events  ({len(ev) / max((t1 - t0), 1e-6) / 1e3:.0f} k/s)")
    if rows:
        path = out / "manifest.csv"
        new = not path.exists()
        with path.open("a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
