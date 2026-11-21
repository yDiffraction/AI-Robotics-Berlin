#!/usr/bin/env python3
"""Procedural throw / feint / idle generator with a simple event-camera simulator.

A stick-figure thrower (capsule limbs) is drawn at `fps` frames per second; an
event simulator converts log-intensity changes into ON/OFF events like the GenX320
(per-pixel contrast threshold with mismatch, timestamps interpolated within a frame,
background activity noise). Output files use the Durance .bin format.

Views:
  profile  camera sees the thrower from the side (like dataset/01 recordings)
  frontal  thrower faces the camera (the robot's view); the ball grows as it approaches

Classes: 01 throw (ball released), 02 feint (same wind-up, arm stops, ball stays in hand),
         03 idle (standing, swaying, waving, arm raised without a throw, walking past).

  uv run tools/synth_throw.py --out dataset/build/synth --count 300 --seed 1
  uv run tools/synth_throw.py --out /tmp/s --count 3 --video /tmp/s/videos   # with MP4 previews
"""

import argparse
import csv
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SENSOR = 320
SS = 2                      # supersampling factor for smooth edges
DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
CLASSES = {"throw": 1, "feint": 2, "idle": 3}


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


class Scene:
    """Randomised thrower, ball, background and camera parameters for one clip."""

    def __init__(self, rng, label, view, duration):
        self.rng, self.label, self.view, self.duration = rng, label, view, duration
        r = rng.uniform
        self.height = r(150, 290)                         # body height in pixels (distance)
        self.cx = r(0.3, 0.7) * SENSOR
        self.ground = r(0.85, 1.05) * SENSOR
        self.facing = rng.choice([-1, 1])                 # profile: throws left or right
        self.background = r(0.35, 0.65)
        sign = 1 if rng.random() < 0.5 else -1
        self.person_level = float(np.clip(self.background + sign * r(0.25, 0.45), 0.05, 0.98))
        self.stripe_level = float(np.clip(self.person_level + rng.choice([-1, 1]) * r(0.1, 0.3), 0.03, 1.0))
        self.ball_level = r(0.05, 0.95)
        self.ball_radius = r(0.022, 0.04) * self.height
        self.limb = r(0.045, 0.07) * self.height
        self.sway = r(1.0, 8.0)
        self.sway_hz = r(0.3, 1.2)
        self.lean_amount = r(0.05, 0.25)
        self.camera_shake = r(0.0, 1.5) if rng.random() < 0.4 else 0.0
        # timeline (seconds)
        self.t_start = r(0.15, 0.35)                      # wind-up starts
        self.windup = r(0.25, 0.45)
        self.hold = r(0.0, 0.18)
        self.forward = r(0.09, 0.17)                      # forward swing to release
        self.t_release = self.t_start + self.windup + self.hold + self.forward
        self.feint_fraction = r(0.0, 0.6)                 # feint: how far the forward swing goes
        self.feint_return = r(0.3, 0.6)
        self.ball_speed = r(4.0, 9.0)                     # m/s (frontal looming)
        self.distance_m = r(2.0, 4.5)
        self.idle_kind = rng.choice(["stand", "wave", "raise", "walk", "reach"])
        self.texture = self._texture()

    def _texture(self):
        """Static background with smooth blobs and a few edges (produces events only on shake)."""
        n = SENSOR * SS
        base = np.full((n, n), self.background, np.float32)
        yy, xx = np.mgrid[0:n, 0:n] / n
        for _ in range(self.rng.integers(2, 6)):
            cx, cy, s = self.rng.uniform(0, 1, 2).tolist() + [self.rng.uniform(0.05, 0.3)]
            base += self.rng.uniform(-0.15, 0.15) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
        for _ in range(self.rng.integers(0, 4)):
            x0 = int(self.rng.uniform(0, n))
            base[:, x0:x0 + int(self.rng.uniform(2, 12))] += self.rng.uniform(-0.2, 0.2)
        return np.clip(base, 0.05, 1.0)

    # --- kinematics -----------------------------------------------------------
    def arm_angles(self, t):
        """Shoulder and elbow angles (radians, 0 = arm hanging down, + = towards throw side back/up)."""
        rest_s, rest_e = 0.15, 0.2
        cocked_s, cocked_e = 2.6, 1.9                     # arm up and back, elbow bent
        release_s, release_e = 4.4, 0.3                   # arm forward over the head
        follow_s, follow_e = 5.9, 0.5                     # follow-through down in front
        t1 = self.t_start + self.windup
        t2 = t1 + self.hold
        t3 = t2 + self.forward
        if self.label == "idle":
            return self.idle_angles(t, rest_s, rest_e)
        if t < self.t_start:
            return rest_s, rest_e
        if t < t1:
            u = smoothstep((t - self.t_start) / self.windup)
            return rest_s + (cocked_s - rest_s) * u, rest_e + (cocked_e - rest_e) * u
        if t < t2:
            return cocked_s, cocked_e
        if self.label == "throw":
            if t < t3:
                u = ((t - t2) / self.forward) ** 1.6       # accelerating
                return cocked_s + (release_s - cocked_s) * u, cocked_e + (release_e - cocked_e) * u
            u = smoothstep((t - t3) / 0.25)
            return release_s + (follow_s - release_s) * u, release_e + (follow_e - release_e) * u
        # feint: start the forward swing, stop at feint_fraction, then bring the arm back down
        stop_s = cocked_s + (release_s - cocked_s) * self.feint_fraction
        stop_e = cocked_e + (release_e - cocked_e) * self.feint_fraction
        t_stop = t2 + self.forward * max(self.feint_fraction, 0.05) * 1.6
        if t < t_stop:
            u = smoothstep((t - t2) / (t_stop - t2))
            return cocked_s + (stop_s - cocked_s) * u, cocked_e + (stop_e - cocked_e) * u
        u = smoothstep((t - t_stop - 0.1) / self.feint_return)
        return stop_s + (rest_s - stop_s) * u, stop_e + (rest_e - stop_e) * u

    def idle_angles(self, t, rest_s, rest_e):
        k = self.idle_kind
        if k == "wave":
            return 2.9, 0.8 + 0.6 * math.sin(2 * math.pi * 2.2 * t)
        if k == "raise":
            u = smoothstep((t - self.t_start) / 0.5) - smoothstep((t - self.t_start - 0.9) / 0.5)
            return rest_s + 2.4 * u, rest_e + 0.4 * u
        if k == "reach":
            u = smoothstep((t - self.t_start) / 0.4) - smoothstep((t - self.t_start - 0.7) / 0.4)
            return rest_s + 1.4 * u, rest_e + 0.1 * u
        return rest_s + 0.1 * math.sin(2 * math.pi * 0.7 * t), rest_e

    def lean(self, t):
        """Torso lean (radians): back during the wind-up, forward during the throw."""
        if self.label == "idle":
            return 0.03 * math.sin(2 * math.pi * self.sway_hz * t)
        t1 = self.t_start + self.windup
        back = smoothstep((t - self.t_start) / self.windup) * self.lean_amount
        fwd_end = 1.0 if self.label == "throw" else self.feint_fraction
        fwd = smoothstep((t - t1 - self.hold) / (self.forward + 0.15)) * self.lean_amount * (1 + fwd_end)
        settle = smoothstep((t - self.t_release - 0.2) / 0.4) * self.lean_amount * fwd_end
        return -back + fwd - settle

    def body_offset(self, t):
        dx = self.sway * math.sin(2 * math.pi * self.sway_hz * t)
        if self.label == "idle" and self.idle_kind == "walk":
            dx += (t - self.duration / 2) * self.rng_walk_speed * SENSOR
        return dx

    @property
    def rng_walk_speed(self):
        if not hasattr(self, "_walk"):
            self._walk = self.rng.choice([-1, 1]) * self.rng.uniform(0.25, 0.6)
        return self._walk

    # --- drawing --------------------------------------------------------------
    def draw(self, t):
        n = SENSOR * SS
        img = Image.fromarray((self.texture * 255).astype(np.uint8))
        if self.camera_shake:
            dx = int(self.camera_shake * SS * math.sin(2 * math.pi * 7.0 * t))
            dy = int(self.camera_shake * SS * math.cos(2 * math.pi * 5.0 * t))
            img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
        d = ImageDraw.Draw(img)
        h = self.height * SS
        cx = (self.cx + self.body_offset(t)) * SS
        gy = self.ground * SS
        col = int(self.person_level * 255)
        w = max(2, int(self.limb * SS))
        hip = (cx, gy - 0.5 * h)
        lean = self.lean(t) * (self.facing if self.view == "profile" else 0.3)
        shoulder = (cx + 0.3 * h * math.sin(lean), gy - 0.5 * h - 0.3 * h * math.cos(lean))
        head = (cx + 0.41 * h * math.sin(lean), gy - 0.5 * h - 0.41 * h * math.cos(lean))
        step = 0.08 * h * (math.sin(2 * math.pi * 1.6 * t) if self.label == "idle" and self.idle_kind == "walk" else 0.3)
        for foot_dx in (-step, step):
            d.line([hip, (cx + foot_dx, gy)], fill=col, width=int(w * 1.3))
        d.line([hip, shoulder], fill=col, width=int(w * 2.2))
        for k in (0.3, 0.55, 0.8):                         # clothing edges
            px = hip[0] + (shoulder[0] - hip[0]) * k
            py = hip[1] + (shoulder[1] - hip[1]) * k
            d.line([(px - w * 1.1, py), (px + w * 1.1, py)], fill=int(self.stripe_level * 255), width=max(1, w // 3))
        r_head = 0.065 * h
        d.ellipse([head[0] - r_head, head[1] - r_head, head[0] + r_head, head[1] + r_head], fill=col)

        s_ang, e_ang = self.arm_angles(t)
        upper, fore = 0.19 * h, 0.17 * h
        if self.view == "profile":
            sgn = -self.facing                               # arm swings back to the side opposite the throw
            elbow = (shoulder[0] + sgn * upper * math.sin(s_ang), shoulder[1] + upper * math.cos(s_ang))
            a2 = s_ang + e_ang
            hand = (elbow[0] + sgn * fore * math.sin(a2), elbow[1] + fore * math.cos(a2))
        else:
            # frontal: the arm moves mostly in depth; project with foreshortening, offset to the side
            side = self.facing * 0.12 * h
            depth = math.sin(s_ang)                          # + = behind the body, - = towards camera
            lift = -math.cos(s_ang)
            elbow = (shoulder[0] + side, shoulder[1] - upper * lift)
            a2 = s_ang + e_ang
            hand = (elbow[0] + side * 0.4 * (1 + depth), elbow[1] - fore * (-math.cos(a2)))
        d.line([shoulder, elbow], fill=col, width=w)
        d.line([elbow, hand], fill=col, width=w)
        # passive arm
        d.line([shoulder, (shoulder[0] - 0.05 * h * (self.facing if self.view == "profile" else 1), hip[1])],
               fill=col, width=w)

        ball = self.ball_position(t, hand)
        if ball is not None:
            bx, by, br = ball
            bcol = int(self.ball_level * 255)
            d.ellipse([bx - br, by - br, bx + br, by + br], fill=bcol)
        small = img.resize((SENSOR, SENSOR), Image.BOX)
        return np.asarray(small, np.float32) / 255.0

    def ball_position(self, t, hand):
        r = self.ball_radius * SS
        if self.label != "throw" or t < self.t_release:
            if self.label == "idle" and self.idle_kind in ("stand", "walk"):
                return None
            return hand[0], hand[1], r
        dt = t - self.t_release
        if self.view == "profile":
            v = self.ball_speed / self.distance_m * self.height * SS * 0.55   # px/s across the image
            x = self.hand_at_release(hand)[0] + self.facing * v * dt
            y = self.hand_at_release(hand)[1] - 0.6 * v * dt + 0.5 * 9.81 * (self.height * SS / 1.75) * dt * dt
            return x, y, r
        # frontal: ball approaches the camera -> grows and moves towards the image centre
        z = self.distance_m - self.ball_speed * dt
        if z <= 0.25:
            return None
        grow = self.distance_m / z
        hx, hy = self.hand_at_release(hand)
        target = (SENSOR * SS / 2, SENSOR * SS * 0.6)
        u = 1 - z / self.distance_m
        return hx + (target[0] - hx) * u, hy + (target[1] - hy) * u, r * grow

    def hand_at_release(self, hand):
        if not hasattr(self, "_release_hand"):
            self._release_hand = hand
        return self._release_hand


def simulate_events(frames, fps, rng, threshold=(0.15, 0.3), mismatch=0.03, noise_hz=(2_000, 40_000)):
    """Convert intensity frames [N, H, W] into events (contrast threshold on log intensity)."""
    c_on = rng.uniform(*threshold) + rng.normal(0, mismatch, frames.shape[1:])
    c_off = c_on * rng.uniform(0.9, 1.1)
    c_on, c_off = np.clip(c_on, 0.05, None), np.clip(c_off, 0.05, None)
    ref = np.log(frames[0] + 0.02)
    dt_us = 1e6 / fps
    xs, ys, ps, ts = [], [], [], []
    for i in range(1, len(frames)):
        cur = np.log(frames[i] + 0.02)
        diff = cur - ref
        n_on = np.floor(np.maximum(diff, 0) / c_on).astype(np.int32)
        n_off = np.floor(np.maximum(-diff, 0) / c_off).astype(np.int32)
        for n, pol, c in ((n_on, 1, c_on), (n_off, 0, -c_off)):
            yy, xx = np.nonzero(n)
            if len(yy) == 0:
                continue
            counts = np.minimum(n[yy, xx], 4)
            xr, yr = np.repeat(xx, counts), np.repeat(yy, counts)
            xs.append(xr); ys.append(yr)
            ps.append(np.full(len(xr), pol, np.uint8))
            ts.append(((i - 1) + rng.random(len(xr))) * dt_us)
            ref[yy, xx] += n[yy, xx] * c[yy, xx]
    duration_us = len(frames) * dt_us
    m = rng.poisson(rng.uniform(*noise_hz) * duration_us / 1e6)
    xs.append(rng.integers(0, SENSOR, m)); ys.append(rng.integers(0, SENSOR, m))
    ps.append(rng.integers(0, 2, m).astype(np.uint8)); ts.append(rng.uniform(0, duration_us, m))
    ev = np.empty(sum(len(a) for a in xs), DTYPE)
    ev["x"], ev["y"] = np.concatenate(xs), np.concatenate(ys)
    ev["polarity"], ev["timestamp"] = np.concatenate(ps), np.concatenate(ts).astype(np.int64)
    return ev[np.argsort(ev["timestamp"], kind="stable")]


def save_bin(path, ev):
    body = ev.astype(DTYPE).tobytes()
    Path(path).write_bytes(struct.pack("!I", len(body)) + body)


def make_clip(rng, label, view, duration, fps):
    scene = Scene(rng, label, view, duration)
    times = np.arange(int(duration * fps)) / fps
    frames = np.stack([scene.draw(t) for t in times])
    ev = simulate_events(frames, fps, rng)
    meta = {"label": label, "view": view, "release_us": int(scene.t_release * 1e6),
            "windup_start_us": int(scene.t_start * 1e6), "height_px": round(scene.height, 1),
            "idle_kind": scene.idle_kind if label == "idle" else "",
            "feint_fraction": round(scene.feint_fraction, 2) if label == "feint" else "",
            "camera_shake": round(scene.camera_shake, 2)}
    return ev, frames, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=30, help="clips per class")
    p.add_argument("--classes", default="throw,feint,idle")
    p.add_argument("--views", default="profile,frontal")
    p.add_argument("--duration", type=float, default=1.3)
    p.add_argument("--fps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="synth")
    p.add_argument("--video", default=None, help="also write MP4 previews (source + events) here")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    views = args.views.split(",")
    rows = []
    for label in args.classes.split(","):
        for k in range(args.count):
            view = views[k % len(views)]
            ev, frames, meta = make_clip(rng, label, view, args.duration, args.fps)
            name = f"gesture_{CLASSES[label]:02d}-{label}-{args.prefix}{args.seed}-{view}-{k:04d}.bin"
            save_bin(out / name, ev)
            meta.update(file=name, events=len(ev))
            rows.append(meta)
            if args.video:
                import events_video
                events_video.write_clip_video(ev, Path(args.video) / name.replace(".bin", ".mp4"),
                                              label=label, release_us=meta["release_us"],
                                              source_frames=frames, source_fps=args.fps)
            print(f"{name}  {len(ev)} events")
    with (out / f"manifest-{args.prefix}{args.seed}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / f"params-{args.prefix}{args.seed}.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
