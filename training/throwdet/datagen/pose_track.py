#!/usr/bin/env python3
"""Person crop + automatic decision time for phone videos, using MediaPipe pose landmarks.

Runs in its own environment (MediaPipe needs a different numpy than riverops):
  uv run --no-project --python 3.12 --with mediapipe --with opencv-python-headless \
      tools/pose_track.py --takes recording/videos/takes.csv --out recording/videos/takes_pose.csv

For every take it tracks the body (33 landmarks per frame) and writes a copy of the CSV with
  crop        square "x,y,size" around the person over the whole take (+ margin)
  release_s   if empty and label != idle: time of peak speed of the faster wrist (the throwing
              hand) -- the same convention as the real camera labels (peak of the arm swing)
  hand        left / right throwing hand, pose_ok (fraction of frames with a detected person)
and one JSON per take with the wrist/elbow/shoulder tracks (for pose-driven synthesis).
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
JOINTS = {"nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
          "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24, "l_ankle": 27, "r_ankle": 28}


def track(path, start, end):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
                                        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    frames = []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if (start and t < start) or (end and t > end):
                continue
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), int(t * 1000))
            if res.pose_landmarks:
                pts = res.pose_landmarks[0]
                frames.append({"t": t, "pts": [(p.x * w, p.y * h, p.visibility) for p in pts]})
            else:
                frames.append({"t": t, "pts": None})
    return frames, fps, w, h


def analyse(frames, w, h, margin=1.25):
    ok = [f for f in frames if f["pts"]]
    if not ok:
        return None
    allp = np.array([[p[0], p[1]] for f in ok for p in f["pts"] if p[2] > 0.3])
    x0, y0 = np.percentile(allp, 1, axis=0)
    x1, y1 = np.percentile(allp, 99, axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(min(max(x1 - x0, y1 - y0) * margin, w, h))
    x = int(np.clip(cx - side / 2, 0, w - side))
    y = int(np.clip(cy - side / 2, 0, h - side))
    t = np.array([f["t"] for f in ok])
    speeds = {}
    for hand, j in (("left", JOINTS["l_wrist"]), ("right", JOINTS["r_wrist"])):
        p = np.array([[f["pts"][j][0], f["pts"][j][1]] for f in ok])
        v = np.hypot(*np.diff(p, axis=0).T) / np.maximum(np.diff(t), 1e-6)
        speeds[hand] = np.convolve(v, np.ones(3) / 3, mode="same") if len(v) >= 3 else v
    hand = max(speeds, key=lambda k: speeds[k].max() if len(speeds[k]) else 0)
    k = int(np.argmax(speeds[hand])) if len(speeds[hand]) else 0
    release = float(t[min(k + 1, len(t) - 1)])
    return {"crop": f"{x},{y},{side}", "release_s": round(release, 3), "hand": hand,
            "pose_ok": round(len(ok) / max(len(frames), 1), 2),
            "peak_wrist_px_s": round(float(speeds[hand][k]), 1) if len(speeds[hand]) else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--takes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(args.takes).parent
    rows = list(csv.DictReader(open(args.takes)))
    tracks_dir = Path(args.out).parent / "pose_tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        video = Path(r["video"]) if Path(r["video"]).is_absolute() else base / r["video"]
        start = float(r["start_s"]) if r.get("start_s") else None
        end = float(r["end_s"]) if r.get("end_s") else None
        frames, fps, w, h = track(video, start, end)
        a = analyse(frames, w, h)
        if a is None:
            print(f"{r['video']}: no person found")
            r.update(pose_ok=0)
            continue
        if not r.get("crop"):
            r["crop"] = a["crop"]
        if r["label"] != "idle" and not r.get("release_s"):
            r["release_s"] = a["release_s"]          # already in the video timeline
            r["release_source"] = "pose_wrist_peak"
        r.update(hand=a["hand"], pose_ok=a["pose_ok"])
        (tracks_dir / f"{i:04d}-{Path(r['video']).stem}.json").write_text(json.dumps(
            {"video": r["video"], "fps": fps, "size": [w, h],
             "tracks": [{"t": f["t"], **({n: f["pts"][j][:2] for n, j in JOINTS.items()} if f["pts"] else {})}
                        for f in frames]}))
        print(f"{r['video']}: crop {r['crop']}  release {r.get('release_s')}  hand {a['hand']}  "
              f"pose_ok {a['pose_ok']}  peak wrist {a['peak_wrist_px_s']} px/s")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
