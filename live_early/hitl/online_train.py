#!/usr/bin/env python3
"""Online trainer for live_loop.py (runs on the GPU machine, started by live_loop.py over SSH).

Trains continuously on the sessions that live_loop.py uploads into <root>/sessions/<stamp>/ while they grow:
  * every --reload-s seconds: re-read all complete chunks + events (presses), and the newest control.json
    (theta set by the person in the window; the RL reward and the event losses use it)
  * one training step after another (train.Trainer; default loss pg = REINFORCE on the alarm rule)
  * every --snapshot-s seconds: export the EMA weights to <root>/snaps/<step>/ (board folder) and write
    <root>/latest.json. live_loop.py decides when (or whether) a snapshot goes onto the board.
  * <root>/status.json every few steps (step, data, loss parts, theta)
No validation, no theta selection: the person sets theta.

  python online_train.py --root live/<id> --loss pg [train.py options, e.g. --bs 16 --lr 3e-4 --ctx-frames ...]
"""
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import train  # noqa: E402
from export_board import export, load_ckpt  # noqa: E402


def write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def main():
    a_root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else None
    try:
        loop()
    except BaseException as e:                         # make a crash visible in the window, not a frozen status
        if a_root:
            write_json(Path(a_root) / "status.json", {"state": f"CRASHED: {type(e).__name__}: {e}"[:300], "t": time.time()})
        raise


def loop():
    ap = train.build_parser()
    ap.description = __doc__
    ap.add_argument("--root", required=True)
    ap.add_argument("--reload-s", type=float, default=30)
    ap.add_argument("--snapshot-s", type=float, default=120)
    ap.add_argument("--keep-snaps", type=int, default=6)
    ap.add_argument("--min-frames", type=int, default=1000, help="wait for this much data before training")
    ap.add_argument("--idle-exit-s", type=float, default=900, help="exit when no upload arrived for this long")
    ap.set_defaults(loss="pg", minutes=1e9)
    a = ap.parse_args()
    losses = set(a.loss.split(","))
    train.setup(a)                                     # model folder -> readout rule, init checkpoint, theta
    root = Path(a.root)
    (root / "snaps").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    model, ck = load_ckpt(a.init)
    model.to(dev)
    trainer = train.Trainer(model, a, losses)
    ctx_segs = [train.segment(s, 0, len(s["frames"]), 0, s["hn"], a)
                for s in (train.load_ctx(Path(a.ctx_frames).expanduser()) if a.ctx_frames else [])]

    sampler, info, step, t_reload, t_snap, parts = None, {}, 0, 0.0, time.time(), {}
    t0 = time.time()
    while True:
        now = time.time()
        last_upload = max([t0] + [f.stat().st_mtime for f in root.glob("sessions/*/*")])
        if now - last_upload > a.idle_exit_s:              # live_loop gone (laptop off, network down for good)
            write_json(root / "status.json", {"state": f"stopped: no upload for {a.idle_exit_s:.0f} s", "step": step, "t": now})
            print("idle exit", flush=True)
            return
        if now - t_reload >= a.reload_s:
            t_reload = now
            ctrl = sorted(root.glob("sessions/*/control.json"), key=lambda p: p.stat().st_mtime)
            if ctrl:
                try:
                    a.mil_theta = float(json.loads(ctrl[-1].read_text())["theta"])
                except (ValueError, KeyError, json.JSONDecodeError):
                    pass
            segs = []
            for d in sorted((root / "sessions").glob("*")):
                try:
                    s = train.load_session(d)
                except Exception as e:         # a chunk mid-upload; next reload gets it
                    print("skip", d.name, e, flush=True)
                    continue
                if s is None or len(s["frames"]) < a.T + a.warm:
                    continue
                segs.append(train.segment(s, 0, len(s["frames"]), a.lag,
                                          train.unclaimed_alarms(s["alarm"], s["presses"], a, a.lag), a))
            human_frames = sum(int((s["w"] > 0).sum()) for s in segs)    # labelled frames (not "just watching")
            info = {"human_min": round(human_frames / 6000, 2), "presses": sum(len(s["windows"]) for s in segs),
                    "hard_neg": sum(len(s["hn"]) for s in segs), "theta": a.mil_theta}
            sampler = train.Sampler(segs + ctx_segs, a, dev, rng) if human_frames >= a.min_frames else None
        if sampler is None:
            write_json(root / "status.json", {"state": "waiting for data", "step": step, "t": time.time(), **info})
            time.sleep(5)
            continue
        parts = trainer.step(sampler)
        step += 1
        if step % 5 == 0:
            write_json(root / "status.json", {"state": "training", "step": step, "t": time.time(),
                                              "min": round((time.time() - t0) / 60, 1), "loss": a.loss, **info, **parts})
            print(json.dumps({"step": step, **parts, **info}), flush=True)
        if time.time() - t_snap >= a.snapshot_s:
            t_snap = time.time()
            net = copy.deepcopy(trainer.avg).cpu().eval()
            d = root / "snaps" / f"{step:06d}"
            md5, _ = export(net, d, a.mil_theta, {"step": step, "loss": a.loss, **info}, model_dir=a.model_dir)
            torch.save({"model": net.state_dict(), "config": {**ck["config"], "online": {"step": step, **info}}}, d / "model.pt")
            write_json(root / "latest.json", {"step": step, "dir": d.name, "md5": md5, "t": time.time(), **info})
            print("snapshot", d, md5, flush=True)
            for old in sorted((root / "snaps").glob("*"))[:-a.keep_snaps]:
                shutil.rmtree(old, ignore_errors=True)


if __name__ == "__main__":
    main()
