#!/usr/bin/env python3
"""Inspect, preview and cut GenX320 event recordings (.bin, Durance dataset format).

File format: 4-byte big-endian payload size, then records of
  x: uint16 BE, y: uint16 BE, polarity: uint8, timestamp: int64 BE (microseconds).

Commands:
  info     print duration, event count and event rate for each file
  preview  write a PNG per file: event-rate curve + accumulated frames over time
  gif      write an animated GIF of the recording (2 channels: ON green, OFF magenta)
  cut      split a long recording into clips around motion bursts (e.g. each throw)

Examples:
  uv run tools/events_tool.py info recordings/s01/*.bin
  uv run tools/events_tool.py preview recordings/s01/*.bin --out previews/
  uv run tools/events_tool.py cut long_take.bin --out clips/ --pre 0.3 --post 1.0
"""

import argparse
import struct
from pathlib import Path

import numpy as np

EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
SENSOR = 320


def load(path):
    raw = Path(path).read_bytes()
    size = struct.unpack("!I", raw[:4])[0]
    if size != len(raw) - 4 or size % EVENT_DTYPE.itemsize:
        raise ValueError(f"{path}: header says {size} bytes, file has {len(raw) - 4}")
    return np.frombuffer(raw, dtype=EVENT_DTYPE, offset=4)


def save(path, events):
    body = events.astype(EVENT_DTYPE).tobytes()
    Path(path).write_bytes(struct.pack("!I", len(body)) + body)


def rate_curve(events, bin_us):
    t = events["timestamp"].astype(np.int64)
    t = t - t[0]
    bins = np.arange(0, t[-1] + bin_us, bin_us)
    on = np.histogram(t[events["polarity"] == 1], bins)[0]
    off = np.histogram(t[events["polarity"] == 0], bins)[0]
    return bins[:-1] / 1e6, on, off


def accumulate(events, t0, t1, downsample=1):
    t = events["timestamp"].astype(np.int64)
    sel = events[(t >= t0) & (t < t1)]
    size = SENSOR // downsample
    img = np.zeros((2, size, size), np.float32)
    np.add.at(img, (sel["polarity"].astype(int).clip(0, 1),
                    sel["y"].astype(int) // downsample,
                    sel["x"].astype(int) // downsample), 1)
    return img


def to_rgb(img):
    scale = max(np.percentile(img[img > 0], 99), 1) if (img > 0).any() else 1
    on, off = np.clip(img[1] / scale, 0, 1), np.clip(img[0] / scale, 0, 1)
    return np.stack([off, on, off], axis=-1)


def cmd_info(args):
    for path in args.files:
        ev = load(path)
        if len(ev) == 0:
            print(f"{path}: EMPTY")
            continue
        dur = (int(ev["timestamp"][-1]) - int(ev["timestamp"][0])) / 1e6
        on = int((ev["polarity"] == 1).sum())
        print(f"{path}: {len(ev):>9} events  {dur:7.3f} s  {len(ev) / max(dur, 1e-9) / 1e6:6.2f} Mev/s"
              f"  ON {on / len(ev):5.1%}  x {ev['x'].min()}-{ev['x'].max()}  y {ev['y'].min()}-{ev['y'].max()}")


def cmd_preview(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        ev = load(path)
        t, on, off = rate_curve(ev, args.bin_ms * 1000)
        start, end = int(ev["timestamp"][0]), int(ev["timestamp"][-1])
        edges = np.linspace(start, end, args.frames + 1)
        fig = plt.figure(figsize=(2.2 * args.frames, 5.2))
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(t, on, color="tab:green", label="ON")
        ax.plot(t, off, color="tab:purple", label="OFF")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"events / {args.bin_ms} ms")
        ax.set_title(Path(path).name)
        ax.legend(loc="upper right")
        for i in range(args.frames):
            a = fig.add_subplot(2, args.frames, args.frames + i + 1)
            a.imshow(to_rgb(accumulate(ev, edges[i], edges[i + 1])))
            a.set_title(f"{(edges[i] - start) / 1e6:.2f}-{(edges[i + 1] - start) / 1e6:.2f} s", fontsize=8)
            a.axis("off")
        fig.tight_layout()
        target = out / (Path(path).stem + ".png")
        fig.savefig(target, dpi=80)
        plt.close(fig)
        print(target)


def cmd_gif(args):
    from PIL import Image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        ev = load(path)
        start, end = int(ev["timestamp"][0]), int(ev["timestamp"][-1])
        step = args.frame_ms * 1000
        frames = []
        for t0 in range(start, end, step):
            rgb = (to_rgb(accumulate(ev, t0, t0 + step, args.downsample)) * 255).astype(np.uint8)
            frames.append(Image.fromarray(rgb).resize((320, 320), Image.NEAREST))
        target = out / (Path(path).stem + ".gif")
        frames[0].save(target, save_all=True, append_images=frames[1:],
                       duration=int(args.frame_ms / args.speed), loop=0)
        print(target, f"({len(frames)} frames)")


def cmd_cut(args):
    ev = load(args.file)
    t, on, off = rate_curve(ev, args.bin_ms * 1000)
    rate = on + off
    baseline = np.median(rate)
    threshold = baseline + args.k * (np.median(np.abs(rate - baseline)) + 1)
    active = rate > threshold
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = int(ev["timestamp"][0])
    ts = ev["timestamp"].astype(np.int64)
    clips, i, n = [], 0, len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        onset = t[i]
        j = i
        gap = int(args.min_gap / (args.bin_ms / 1000))
        while j < n and (active[j] or active[j:j + gap].any()):
            j += 1
        c0 = start + int((onset - args.pre) * 1e6)
        c1 = start + int((onset + args.post) * 1e6)
        clip = ev[(ts >= c0) & (ts < c1)]
        if len(clip):
            name = out / f"{Path(args.file).stem}-clip{len(clips) + 1:03d}.bin"
            save(name, clip)
            clips.append((name, onset))
            print(f"{name}  onset {onset:7.3f} s  {len(clip)} events")
        i = max(j, i + int(args.post / (args.bin_ms / 1000)))
    print(f"{len(clips)} clips, threshold {threshold:.0f} events/{args.bin_ms} ms (baseline {baseline:.0f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info")
    s.add_argument("files", nargs="+")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("preview")
    s.add_argument("files", nargs="+")
    s.add_argument("--out", default="previews")
    s.add_argument("--frames", type=int, default=8)
    s.add_argument("--bin-ms", type=float, default=10)
    s.set_defaults(fn=cmd_preview)

    s = sub.add_parser("gif")
    s.add_argument("files", nargs="+")
    s.add_argument("--out", default="previews")
    s.add_argument("--frame-ms", type=int, default=20)
    s.add_argument("--speed", type=float, default=0.5, help="playback speed (0.5 = half speed)")
    s.add_argument("--downsample", type=int, default=2)
    s.set_defaults(fn=cmd_gif)

    s = sub.add_parser("cut")
    s.add_argument("file")
    s.add_argument("--out", default="clips")
    s.add_argument("--pre", type=float, default=0.3, help="seconds kept before motion onset")
    s.add_argument("--post", type=float, default=1.0, help="seconds kept after motion onset")
    s.add_argument("--bin-ms", type=float, default=10)
    s.add_argument("--k", type=float, default=6.0, help="onset threshold in robust std units")
    s.add_argument("--min-gap", type=float, default=0.5, help="seconds of quiet that end a burst")
    s.set_defaults(fn=cmd_cut)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
