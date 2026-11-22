#!/usr/bin/env python3
"""RGB video -> raw DVS events with v2e (SuperSloMo frame interpolation + v2e DVS pixel model).

Runs in the v2e environment (Python 3.14, `pip install v2e`); see README.md.

Per clip:
1. decode grayscale, measure global camera motion (phase correlation between frames). Clips
   whose background moves are rejected: our camera is static, and a panning camera turns the
   whole background into events, which no GenX320 clip of ours has.
2. pick the window: motion-energy peak (frame differences) for pos / amb; a random window
   for neg. Window = peak - 1.1 s .. peak + 0.8 s (0.2 s pre-roll for the pixel model).
3. square crop of the full frame height around the motion, resized to 320 x 320. Nothing is
   padded, so there are no empty borders.
4. SuperSloMo x`--up` (30 fps -> 300 fps) so motion is sampled every ~3 ms. Without it every
   10 ms board frame would see events from only every 3rd frame: a 33 ms comb that real
   sensors never show and that a network would pick up immediately.
5. v2e EventEmulator with per-clip sampled thresholds / bandwidth / refractory and no noise.
   Noise is NOT simulated: match.py adds noise measured on our own GenX320 instead.

Output: <out>/<role>/<stem>.npz with t (us), x, y, p (1 = ON), wh, meta (json).

  python v2e_convert.py --manifest data/extpos/hmdb/manifest.csv --videos data/extpos/hmdb/videos \
      --out data/extpos/hmdb/raw --jobs 4 [--limit-per-class 3]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

S = 320
PRE_S, POST_S, PREROLL_S = 0.9, 0.8, 0.2
SLOMO_CKPT = os.environ.get("SLOMO_CKPT")   # default: v2e downloads SuperSloMo39.ckpt to its data folder


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = map(int, (s["avg_frame_rate"] if s["avg_frame_rate"] != "0/0" else s["r_frame_rate"]).split("/"))
    return int(s["width"]), int(s["height"]), num / den


def decode(path, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def camera_motion(fr):
    """Per-frame background displacement (px at 240-px frame height): corner features tracked with
    pyramidal LK, a similarity transform fitted with RANSAC (the static background is the
    inlier majority, the moving person the outliers), displacement of the frame centre.
    Calibrated on HMDB51 camera tags: p90 <= 0.5 keeps 72 % of 'static' and 10 % of 'moving'."""
    import cv2
    sh = []
    for a, b in zip(fr[:-1], fr[1:]):
        p0 = cv2.goodFeaturesToTrack(a, 300, 0.01, 6)
        if p0 is None or len(p0) < 12:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
        ok = st[:, 0] == 1
        if ok.sum() < 12:
            continue
        M, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], ransacReprojThreshold=1.0)
        if M is None:
            continue
        c = np.array([a.shape[1] / 2, a.shape[0] / 2])
        sh.append(float(np.hypot(*(M[:, :2] @ c + M[:, 2] - c))) * 240 / a.shape[0])
    return np.array(sh) if sh else np.array([np.inf])


def motion_energy(fr):
    d = np.abs(np.diff(fr.astype(np.int16), axis=0)) > 15
    e = d.reshape(len(d), -1).mean(1)
    return np.convolve(e, np.ones(3) / 3, "same"), d


class SloMo:
    """SuperSloMo (the model v2e ships) run on in-memory grayscale frames."""

    def __init__(self, up, threads=2):
        import torch
        from v2ecore import model
        from v2ecore.v2e_utils import ensure_slomo_checkpoint
        torch.set_num_threads(threads)
        self.torch, self.up = torch, up
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(ensure_slomo_checkpoint(SLOMO_CKPT), map_location=self.dev, weights_only=False)
        self.fc, self.at = model.UNet(2, 4), model.UNet(12, 5)
        self.fc.load_state_dict(ck["state_dictFC"]); self.at.load_state_dict(ck["state_dictAT"])
        self.fc.to(self.dev).eval(); self.at.to(self.dev).eval()
        self.warp = model.backWarp(S, S, self.dev).to(self.dev)

    def __call__(self, fr, batch=8):
        """fr [N,S,S] uint8 -> [(N-1)*up + 1, S, S] float32 0..255 (input frames included)."""
        torch = self.torch
        x = torch.from_numpy(fr.astype(np.float32) / 255.0)[:, None].to(self.dev)
        out = []
        with torch.no_grad():
            for b0 in range(0, len(fr) - 1, batch):
                I0, I1 = x[b0:b0 + batch], x[b0 + 1:b0 + batch + 1]
                I0 = I0[:len(I1)]
                flow = self.fc(torch.cat((I0, I1), 1))
                F01, F10 = flow[:, :2], flow[:, 2:]
                seq = [I0]
                for k in range(1, self.up):
                    t = k / self.up
                    c = -t * (1 - t)
                    Ft0 = c * F01 + t * t * F10
                    Ft1 = (1 - t) * (1 - t) * F01 + c * F10
                    g0, g1 = self.warp(I0, Ft0), self.warp(I1, Ft1)
                    o = self.at(torch.cat((I0, I1, F01, F10, Ft1, Ft0, g1, g0), 1))
                    Ft0f, Ft1f = o[:, :2] + Ft0, o[:, 2:4] + Ft1
                    V0 = torch.sigmoid(o[:, 4:5]); V1 = 1 - V0
                    seq.append(((1 - t) * V0 * self.warp(I0, Ft0f) + t * V1 * self.warp(I1, Ft1f))
                               / ((1 - t) * V0 + t * V1))
                # interleave: for each pair i: I0_i, interp_1..up-1
                st = torch.stack(seq, 1)                    # [b, up, 1, S, S]
                out.append(st.reshape(-1, S, S))
            out.append(x[-1:, 0])
        return (torch.cat(out).clamp(0, 1) * 255).cpu().numpy()


class DisInterp:
    """CPU fallback: bidirectional DIS optical flow + backward warping (no learned model).
    Coarser than SuperSloMo at occlusions, but removes the 33 ms frame comb just as well."""

    def __init__(self, up):
        import cv2
        self.cv2, self.up = cv2, up
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def __call__(self, fr):
        cv2 = self.cv2
        gy, gx = np.mgrid[0:S, 0:S].astype(np.float32)
        out = []
        for a, b in zip(fr[:-1], fr[1:]):
            f01 = self.dis.calc(a, b, None)
            f10 = self.dis.calc(b, a, None)
            out.append(a.astype(np.float32))
            for k in range(1, self.up):
                t = k / self.up
                w0 = cv2.remap(a, gx + t * f10[..., 0], gy + t * f10[..., 1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                w1 = cv2.remap(b, gx + (1 - t) * f01[..., 0], gy + (1 - t) * f01[..., 1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                out.append((1 - t) * w0 + t * w1)
        out.append(fr[-1].astype(np.float32))
        return np.stack(out)


def simulate(frames, fps_out, rng):
    from v2ecore.emulator import EventEmulator
    c = float(rng.uniform(0.15, 0.25))
    prm = dict(pos_thres=c * float(rng.uniform(0.9, 1.1)), neg_thres=c * float(rng.uniform(0.9, 1.1)),
               sigma_thres=float(rng.uniform(0.02, 0.05)),
               cutoff_hz=float(rng.uniform(0.25, 0.9) * fps_out / (2 * np.pi)),
               leak_rate_hz=0.0, shot_noise_rate_hz=0.0, refractory_period_s=float(rng.uniform(1e-4, 1e-3)))
    import torch
    em = EventEmulator(**prm, device="cuda" if torch.cuda.is_available() else "cpu",
                       seed=int(rng.integers(1 << 30)))
    evs = []
    for i, f in enumerate(frames):
        e = em.generate_events(f.astype(np.float64), i / fps_out)
        if e is not None and len(e):
            evs.append(e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e))
    em.cleanup()
    if not evs:
        return None, prm
    e = np.concatenate(evs)
    return e, prm


def convert(job):
    row, videos, out, up, threads, max_cam, interp = job
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(threads)
    stem = Path(row["file"]).stem
    dst = Path(out) / row["role"] / f"{stem}.npz"
    if dst.exists():
        return {"file": row["file"], "status": "exists"}
    rng = np.random.default_rng(abs(hash(stem)) % (1 << 32))
    t0 = time.time()
    w, h, fps = probe(Path(videos) / row["file"])
    fr = decode(Path(videos) / row["file"], w, h)
    n = len(fr)
    info = {"file": row["file"], "fps": round(fps, 3), "w": w, "h": h, "frames": n}
    if n / fps < 1.4 or fps > 61:
        return {**info, "status": "too_short_or_slowmo"}
    cam = camera_motion(fr)
    info["cam_p90"] = round(float(np.percentile(cam, 90)), 3)
    if info["cam_p90"] > max_cam:
        return {**info, "status": "camera_moves"}
    e, d = motion_energy(fr)
    L = n
    if row["role"] == "neg":
        a = float(rng.uniform(PRE_S + PREROLL_S, max(PRE_S + PREROLL_S, n / fps - POST_S)))
    else:
        lo, hi = int((PRE_S - 0.2) * fps), max(int((PRE_S - 0.2) * fps) + 1, L - int((POST_S - 0.2) * fps))
        a = (lo + int(np.argmax(e[lo:hi]))) / fps if hi > lo else n / fps / 2
    f0 = max(0, int(round((a - PRE_S - PREROLL_S) * fps)))
    f1 = min(n, int(round((a + POST_S) * fps)) + 1)
    fr = fr[f0:f1]
    # square crop (side = frame height) around the motion of this window
    dd = d[f0:max(f0 + 1, f1 - 1)].sum((0, 1)).astype(np.float64)
    side = min(w, h)
    cx = (dd * np.arange(w)).sum() / dd.sum() if dd.sum() > 0 else w / 2
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = (h - side) // 2
    crop = np.stack([cv2.resize(f[y0:y0 + side, x0:x0 + side], (S, S), interpolation=cv2.INTER_AREA)
                     for f in fr])
    hi_fr = (SloMo(up, threads) if interp == "slomo" else DisInterp(up))(crop)
    fps_out = fps * up
    ev, prm = simulate(hi_fr, fps_out, rng)
    if ev is None:
        return {**info, "status": "no_events"}
    t_us = (ev[:, 0] * 1e6).astype(np.int64) + int(round(f0 / fps * 1e6))
    meta = {**info, **{k: row[k] for k in row}, "crop": [x0, y0, side], "anchor_s": round(a, 3),
            "win_s": [round(f0 / fps, 3), round(f1 / fps, 3)], "up": up, "interp": interp, "v2e": prm,
            "sec": round(time.time() - t0, 1)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, t=t_us, x=ev[:, 1].astype(np.int16), y=ev[:, 2].astype(np.int16),
                        p=(ev[:, 3] > 0).astype(np.uint8), wh=np.array([S, S]), meta=json.dumps(meta))
    return {**info, "status": "ok", "events": len(ev), "sec": meta["sec"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--up", type=int, default=10, help="SuperSloMo upsampling factor")
    ap.add_argument("--interp", choices=["slomo", "dis"], default="slomo",
                    help="slomo = SuperSloMo as in v2e (use a GPU); dis = optical-flow CPU fallback")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per job")
    ap.add_argument("--max-cam", type=float, default=0.5, help="max p90 background shift (px/frame at 240 px)")
    ap.add_argument("--static-tag-only", action="store_true", help="skip clips not tagged nm")
    ap.add_argument("--roles", default="pos,neg,amb")
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--skip-bad", action="store_true", help="skip clips tagged qual=bad")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.manifest)) if r["role"] in a.roles.split(",")]
    if a.static_tag_only:
        rows = [r for r in rows if r["cam"] == "nm"]
    if a.skip_bad:
        rows = [r for r in rows if r["qual"] != "bad"]
    if a.limit_per_class:
        rng = np.random.default_rng(0)
        by = {}
        for r in rows:
            by.setdefault(r["cls"], []).append(r)
        rows = [r for c in sorted(by) for r in
                [by[c][i] for i in rng.permutation(len(by[c]))[:a.limit_per_class]]]
    if a.interp == "slomo":
        SloMo(a.up, 1)  # downloads the checkpoint once before the workers start
    log = Path(a.out) / "convert_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(r, a.videos, a.out, a.up, a.threads, a.max_cam, a.interp) for r in rows]
    t0, stat = time.time(), {}
    with ProcessPoolExecutor(a.jobs) as ex, log.open("a") as lf:
        for i, res in enumerate(ex.map(convert, jobs, chunksize=1)):
            lf.write(json.dumps(res) + "\n"); lf.flush()
            stat[res["status"]] = stat.get(res["status"], 0) + 1
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"{i + 1}/{len(jobs)} {stat} {el:.0f}s, eta {el / (i + 1) * (len(jobs) - i - 1):.0f}s",
                      flush=True)


if __name__ == "__main__":
    sys.exit(main())
