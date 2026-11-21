#!/usr/bin/env python3
"""Mass data generation on all CPU cores: procedural clips and phone-video conversions.

Every job has its own seed, so runs are reproducible and can be extended later
(`--start` continues numbering). Each worker writes its .bin file; the parent writes
one manifest row per clip (`manifest.csv` in the output folder, appended).

  # 3000 procedural clips (1000 per class), 14 worker processes, 30 previews
  uv run tools/generate_data.py synth --out dataset/gen/synth --per-class 1000 --workers 14 --previews 30

  # same on a GPU box: CPU workers render, the GPU simulates 32 clips at a time
  uv run tools/generate_data.py synth --out dataset/gen/synth --per-class 10000 --device cuda --batch 32

  # phone videos -> events, 20 variants per take
  uv run tools/generate_data.py video --takes recording/videos/takes.csv --out dataset/gen/video --variants 20 --workers 14
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

CLASS_ID = {"throw": 1, "feint": 2, "idle": 3}


def synth_render(job):
    """CPU part: draw the scene frames (uint8) and draw the per-clip sensor parameters."""
    import event_sim
    import synth_throw as S
    rng = np.random.default_rng(job["seed"])
    scene = S.Scene(rng, job["label"], job["view"], job["pre"] + job["post"] + 0.3)
    times = np.arange(int(scene.duration * job["fps"])) / job["fps"]
    frames = np.stack([scene.draw(t) for t in times])
    meta = {"release_us": int(scene.t_release * 1e6), "windup_start_us": int(scene.t_start * 1e6),
            "height_px": round(scene.height, 1), "camera_shake": round(scene.camera_shake, 2),
            "idle_kind": scene.idle_kind if job["label"] == "idle" else "",
            "feint_fraction": round(scene.feint_fraction, 2) if job["label"] == "feint" else ""}
    return frames, times, event_sim.SensorParams.sample(rng), meta, rng


def synth_finish(job, ev, frames, meta, rng):
    """Cut the training window, save the clip (+ preview), return the manifest row."""
    import synth_throw as S
    label, view = job["label"], job["view"]
    if label != "idle":
        t0 = max(0, meta["release_us"] - int(job["pre"] * 1e6) + int(rng.uniform(-0.1, 0.1) * 1e6))
        rel = meta["release_us"] - t0
    else:
        t0, rel = int(rng.uniform(0, 0.3) * 1e6), None
    t1 = t0 + int((job["pre"] + job["post"]) * 1e6)
    ev = ev[(ev["timestamp"] >= t0) & (ev["timestamp"] < t1)].copy()
    ev["timestamp"] -= t0
    name = f"gesture_{CLASS_ID[label]:02d}-{label}-synth-{view}-{job['index']:06d}.bin"
    S.save_bin(Path(job["out"]) / name, ev)
    if job["preview"]:
        import events_video
        k0 = int(t0 / 1e6 * job["fps"])
        events_video.write_clip_video(ev, Path(job["out"]) / "videos" / name.replace(".bin", ".mp4"), label=label,
                                      release_us=rel, source_frames=frames[k0:].astype(np.float32) / 255,
                                      source_fps=job["fps"])
    return {"file": name, "label": label, "source": "synth", "parent": "", "view": view, "seed": job["seed"],
            "release_us": "" if rel is None else rel, "events": len(ev), "params": json.dumps(meta)}


def synth_job(job):
    import event_sim
    frames, times, params, meta, rng = synth_render(job)
    ev = event_sim.simulate(frames, times * 1e6, params, rng)
    return synth_finish(job, ev, (frames * 255).astype(np.uint8), meta, rng)


def synth_render_u8(job):
    frames, times, params, meta, rng = synth_render(job)
    return job, (np.clip(frames, 0, 1) * 255).astype(np.uint8), times, params, meta, rng


def run_gpu(jobs, out, workers, device, batch):
    """CPU workers render frames, the GPU simulates `batch` clips at once."""
    import event_sim
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.csv"
    new = not path.exists()
    t = time.time()
    done = 0
    with path.open("a", newline="") as f, ProcessPoolExecutor(workers) as pool:
        writer = None
        pending = [pool.submit(synth_render_u8, j) for j in jobs[:max(workers, batch) * 2]]
        queue = jobs[max(workers, batch) * 2:]
        ready = []
        while pending or ready:
            nxt = []
            for fut in pending:
                if fut.done():
                    ready.append(fut.result())
                    if queue:
                        nxt.append(pool.submit(synth_render_u8, queue.pop(0)))
                else:
                    nxt.append(fut)
            pending = nxt
            by_len = {}
            for r in ready:
                by_len.setdefault(len(r[2]), []).append(r)
            group = max(by_len.values(), key=len) if by_len else []
            if len(group) < batch and pending:
                time.sleep(0.05)
                continue
            group = group[:batch]
            ids = {id(r) for r in group}
            ready = [r for r in ready if id(r) not in ids]
            ts = time.time()
            evs = event_sim.simulate_batch(np.stack([r[1] for r in group]), group[0][2] * 1e6,
                                           [r[3] for r in group], group[0][5], device=device)
            tf = time.time()
            for (job, frames, _, _, meta, rng), ev in zip(group, evs):
                row = synth_finish(job, ev, frames, meta, rng)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if new:
                        writer.writeheader()
                writer.writerow(row)
                done += 1
            f.flush()
            rate = done / (time.time() - t) * 3600
            print(f"{done}/{len(jobs)} clips  ({rate:.0f} clips/hour)  sim {tf - ts:.1f}s  save {time.time() - tf:.1f}s  "
                  f"buffered {len(ready)}", flush=True)


def run_pool(jobs, fn, out, workers):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.csv"
    new = not path.exists()
    t = time.time()
    done = 0
    with path.open("a", newline="") as f, ProcessPoolExecutor(workers) as pool:
        writer = None
        futures = [pool.submit(fn, j) for j in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
            done += 1
            if done % 50 == 0 or done == len(jobs):
                rate = done / (time.time() - t) * 3600
                print(f"{done}/{len(jobs)} clips  ({rate:.0f} clips/hour)", flush=True)


def video_job(job):
    """Run video_to_events for a single take (keeps that script the single source of truth)."""
    take_csv = Path(job["out"]) / f".take-{job['index']:04d}.csv"
    with take_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(job["take"].keys()))
        w.writeheader()
        w.writerow(job["take"])
    sub = Path(job["out"]) / f".part-{job['index']:04d}"
    subprocess.run([sys.executable, str(Path(__file__).parent / "video_to_events.py"), "--takes", str(take_csv),
                    "--out", str(sub), "--variants", str(job["variants"]), "--seed", str(job["seed"]),
                    "--previews", str(job["previews"])], check=True, capture_output=True)
    rows = list(csv.DictReader(open(sub / "manifest.csv"))) if (sub / "manifest.csv").exists() else []
    for r in rows:
        os.replace(sub / r["file"], Path(job["out"]) / r["file"])
    vids = sub / "videos"
    if vids.exists():
        (Path(job["out"]) / "videos").mkdir(exist_ok=True)
        for v in vids.iterdir():
            os.replace(v, Path(job["out"]) / "videos" / v.name)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--per-class", type=int, default=300)
    s.add_argument("--classes", default="throw,feint,idle")
    s.add_argument("--views", default="frontal,frontal,profile")
    s.add_argument("--pre", type=float, default=0.7)
    s.add_argument("--post", type=float, default=0.5)
    s.add_argument("--fps", type=int, default=500)
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    s.add_argument("--previews", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--start", type=int, default=0, help="first clip index (to extend an existing set)")
    s.add_argument("--device", default="cpu", help="cpu (numpy per worker) or cuda / mps (batched torch)")
    s.add_argument("--batch", type=int, default=32, help="clips simulated together on the GPU")
    v = sub.add_parser("video")
    v.add_argument("--takes", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--variants", type=int, default=20)
    v.add_argument("--previews", type=int, default=1)
    v.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    v.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "synth":
        views = args.views.split(",")
        jobs = []
        idx = args.start
        for k in range(args.per_class):
            for label in args.classes.split(","):
                jobs.append({"index": idx, "seed": args.seed * 10_000_000 + idx, "label": label,
                             "view": views[k % len(views)], "pre": args.pre, "post": args.post, "fps": args.fps,
                             "out": str(out), "preview": idx - args.start < args.previews})
                idx += 1
        if args.device == "cpu":
            run_pool(jobs, synth_job, out, args.workers)
        else:
            run_gpu(jobs, out, args.workers, args.device, args.batch)
    else:
        takes = list(csv.DictReader(open(args.takes)))
        base = Path(args.takes).parent.resolve()
        for t in takes:
            if not Path(t["video"]).is_absolute():
                t["video"] = str(base / t["video"])
        jobs = [{"index": i, "take": t, "variants": args.variants, "previews": args.previews,
                 "seed": args.seed * 100_000 + i, "out": str(out)} for i, t in enumerate(takes)]
        path = out / "manifest.csv"
        new = not path.exists()
        t0 = time.time()
        with path.open("a", newline="") as f, ProcessPoolExecutor(args.workers) as pool:
            writer = None
            for i, fut in enumerate(as_completed([pool.submit(video_job, j) for j in jobs]), 1):
                rows = fut.result()
                if rows and writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    if new:
                        writer.writeheader()
                if rows:
                    writer.writerows(rows)
                    f.flush()
                print(f"take {i}/{len(jobs)} done, {len(rows)} clips ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
