#!/usr/bin/env python3
"""Render event recordings (.bin) to MP4 videos: single clips and labelled review reels.

Frames accumulate `frame_ms` of events (ON green, OFF magenta). Playback runs at
`speed` x real time (0.25 = four times slower). A header shows the label and time; a
timeline marks the release moment when it is known. Synthetic clips can show the
rendered source image next to the events.

  uv run tools/events_video.py clip dataset/01/samples/x.bin --out x.mp4 --label throw
  uv run tools/events_video.py reel dataset/build/train/*.bin --out reel.mp4 --grid 4x3
"""

import argparse
import csv
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SENSOR = 320
DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
COLORS = {"throw": (255, 80, 60), "feint": (255, 200, 40), "idle": (120, 170, 255)}


def load(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("!I", raw[:4])[0]
    return np.frombuffer(raw, dtype=DTYPE, offset=4, count=n // DTYPE.itemsize)


def label_from_name(path):
    name = Path(path).name
    for k in COLORS:
        if f"-{k}-" in name:
            return k
    return ""


def event_frame(ev, t0, t1, gain=3.0):
    sel = ev[(ev["timestamp"] >= t0) & (ev["timestamp"] < t1)]
    img = np.zeros((SENSOR, SENSOR, 3), np.float32)
    on = sel["polarity"] == 1
    np.add.at(img, (sel["y"][on].astype(int), sel["x"][on].astype(int), 1), 1)
    np.add.at(img, (sel["y"][~on].astype(int), sel["x"][~on].astype(int), 0), 1)
    img[..., 2] = img[..., 0]
    return np.clip(img * (255 / gain), 0, 255).astype(np.uint8)


def font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def annotate(tile, label, t_s, release_s, duration_s, title=""):
    img = Image.fromarray(tile)
    d = ImageDraw.Draw(img)
    color = COLORS.get(label, (230, 230, 230))
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((6, 3), f"{label.upper() or '?'}  {title}", fill=color, font=font(15))
    d.text((img.width - 70, 3), f"{t_s:5.2f}s", fill=(230, 230, 230), font=font(15))
    y = img.height - 8
    d.rectangle([0, y, img.width, img.height], fill=(40, 40, 40))
    d.rectangle([0, y, int(img.width * min(1, t_s / max(duration_s, 1e-6))), img.height], fill=(200, 200, 200))
    if release_s is not None:
        xr = int(img.width * release_s / duration_s)
        d.rectangle([xr - 1, y - 6, xr + 1, img.height], fill=color)
        if release_s <= t_s < release_s + 0.25:
            d.text((6, 28), "RELEASE" if label == "throw" else "DECISION", fill=color, font=font(18))
    return np.asarray(img)


class Writer:
    def __init__(self, path, width, height, fps):
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg not found")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "20", str(path)], stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def write_clip_video(ev, path, label="", release_us=None, source_frames=None, source_fps=None,
                     frame_ms=10, speed=0.25, title=""):
    ev = np.asarray(ev)
    start = int(ev["timestamp"][0]) if len(ev) else 0
    end = int(ev["timestamp"][-1]) if len(ev) else 1
    duration = (end - start) / 1e6
    step = frame_ms * 1000
    fps = max(1, round(1000 / frame_ms * speed))
    width = SENSOR * (2 if source_frames is not None else 1)
    w = Writer(path, width, SENSOR, fps)
    rel = None if release_us is None or release_us == "" else int(release_us) / 1e6
    for t0 in range(start, end, step):
        tile = event_frame(ev, t0, t0 + step)
        t_s = (t0 - start) / 1e6
        tile = annotate(tile, label, t_s, rel, duration, title)
        if source_frames is not None:
            k = min(len(source_frames) - 1, int(t_s * source_fps))
            src = np.repeat((source_frames[k] * 255).astype(np.uint8)[..., None], 3, axis=2)
            tile = np.concatenate([src, tile], axis=1)
        w.write(tile)
    w.close()


def write_reel(files, path, grid=(4, 3), duration_s=1.0, frame_ms=10, speed=0.25, releases=None, scale=0.6):
    cols, rows = grid
    per = cols * rows
    size = int(SENSOR * scale) // 2 * 2
    fps = max(1, round(1000 / frame_ms * speed))
    w = Writer(path, cols * size, rows * size, fps)
    step = frame_ms * 1000
    for page in range(0, len(files), per):
        batch = files[page:page + per]
        clips = [(f, load(f)) for f in batch]
        for k in range(int(duration_s * 1e6 / step)):
            canvas = np.zeros((rows * size, cols * size, 3), np.uint8)
            for i, (f, ev) in enumerate(clips):
                s = int(ev["timestamp"][0]) if len(ev) else 0
                t0 = s + k * step
                rel = (releases or {}).get(Path(f).name)
                tile = annotate(event_frame(ev, t0, t0 + step), label_from_name(f), k * step / 1e6,
                                None if rel in (None, "") else int(rel) / 1e6, duration_s,
                                Path(f).stem.split("-", 2)[-1][:22])
                tile = np.asarray(Image.fromarray(tile).resize((size, size)))
                r, c = divmod(i, cols)
                canvas[r * size:(r + 1) * size, c * size:(c + 1) * size] = tile
            w.write(canvas)
    w.close()


def read_releases(manifests):
    out = {}
    for m in manifests:
        with open(m) as f:
            for row in csv.DictReader(f):
                out[row["file"]] = row.get("release_us", "")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clip")
    c.add_argument("file")
    c.add_argument("--out", required=True)
    c.add_argument("--label", default=None)
    c.add_argument("--release-us", default=None)
    c.add_argument("--speed", type=float, default=0.25)
    r = sub.add_parser("reel")
    r.add_argument("files", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--grid", default="4x3")
    r.add_argument("--duration", type=float, default=1.0)
    r.add_argument("--speed", type=float, default=0.25)
    r.add_argument("--manifest", nargs="*", default=[])
    args = p.parse_args()
    if args.cmd == "clip":
        write_clip_video(load(args.file), args.out, args.label or label_from_name(args.file),
                         args.release_us, speed=args.speed)
    else:
        cols, rows = map(int, args.grid.split("x"))
        write_reel(sorted(args.files), args.out, (cols, rows), args.duration, speed=args.speed,
                   releases=read_releases(args.manifest))
    print(args.out)


if __name__ == "__main__":
    main()
