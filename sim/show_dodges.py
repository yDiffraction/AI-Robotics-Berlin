"""3D viewer: several Go2s side by side, each doing a different sideways dodge at the same moment.

All dodges are simulated once at start-up (a few seconds) and recorded at the 2 ms sim rate;
the viewer then plays the recording back, so any speed is smooth and you can scrub freely.

Layout: one lane per dog (rows along x, two columns along y), all dodging towards +y = screen
right. Floor: white start line, 30 cm line (red, turns green when that dog crosses it), and a colour
patch matching the ball on the dog (the ball turns black if the dog falls). Live times and results
are in the control panel; the 3D overlay text is a static legend (MuJoCo's set_texts costs ~0.4 s per
call, so it must not be updated per frame).

Controls (control panel window):  time slider, speed slider, play/pause, step, restart, fit view, loop
Keys (in the 3D window):          space play/pause · ←/→ step 10 ms · ↑/↓ speed · R restart · F fit view
Mouse (3D window):                left-drag rotate · right-drag pan · scroll zoom

  python sim/show_dodges.py                      # opens the 3D window + control panel
  python sim/show_dodges.py --speed 0.25         # start slower
  python sim/show_dodges.py --snapshot out.png --at 0.4   # no window: one rendered frame
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import time
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch

import go2_dodge_sim as g

HERE = Path(__file__).resolve().parent
ROW = 1.25                     # row spacing along x [m]
COL = 2.6                      # column spacing along y [m] (a dog that runs away covers up to ~1.8 m)
COLORS = [(0.95, 0.25, 0.25, 1), (1.0, 0.6, 0.1, 1), (0.95, 0.9, 0.2, 1), (0.3, 0.8, 0.3, 1),
          (0.25, 0.6, 1.0, 1), (0.7, 0.4, 1.0, 1), (1.0, 1.0, 1.0, 1)]
COLOR_NAMES = ["red", "orange", "yellow", "green", "blue", "purple", "white"]
LEAD_IN = 0.5                  # seconds shown before the dodge command
LINE_RED, LINE_GREEN, FELL_RGBA = (1, 0.1, 0.1, 0.95), (0.1, 1.0, 0.2, 1.0), (0.05, 0.05, 0.05, 1)


def optimized_dodges() -> list[g.Dodge]:
    """The two best finalists of optimize_dodge.py: fastest, and lowest tilt among the fully robust ones."""
    path = HERE / "results/optimize.json"
    if not path.exists():
        return []
    import optimize_dodge as o
    final = json.loads(path.read_text())["final"]
    robust = [f for f in final if f["validation"]["ok"] == f["validation"]["n"] and f["neighbours_ok"] == f["neighbours_n"]]
    if not robust:
        return []
    fast = min(robust, key=lambda f: (f["search"]["t30_nominal"], f["validation"]["t30_worst"]))
    steady = min(robust, key=lambda f: f["validation"]["tilt_worst"])
    out = [o.to_dodge(fast["search"]["params"], "OPTIMIZED fastest")]
    if steady is not fast:
        out.append(o.to_dodge(steady["search"]["params"], "OPTIMIZED steadiest (least tilt)"))
    return out


SAFE_NAME = "SAFE (load-limited, 0.1 rad joint clamp)"
SAFE_PHYS = g.Physics(target_margin=0.1)     # the SAFE dodge runs with the deploy-script joint clamp


def safe_dodge() -> g.Dodge | None:
    """The SAFE dodge: chosen by `validate_dodge.py --scale-sweep` (results/safe_dodge.json), else the best
    finalist of `optimize_dodge.py --safe` that had no failure at all."""
    import optimize_dodge as o
    chosen = HERE / "results/safe_dodge.json"
    if chosen.exists():
        return o.to_dodge(json.loads(chosen.read_text())["chosen"]["params"], SAFE_NAME)
    path = HERE / "results/optimize_safe.json"
    if not path.exists():
        return None
    final = json.loads(path.read_text())["final"]
    ok = [f for f in final if f["validation"]["ok"] == f["validation"]["n"] and f["neighbours_ok"] == f["neighbours_n"]]
    if not ok:
        return None
    best = min(ok, key=lambda f: (f["validation"]["t30_worst"], f["validation"]["tilt_worst"]))
    return o.to_dodge(best["search"]["params"], SAFE_NAME)


def lineup() -> list[g.Dodge]:
    safe = safe_dodge()
    return [
        g.Dodge("trained limit: vy 0.6, from standstill", g.Command(vy=0.6), from_stand=True),
        g.Dodge("sustained vy 1.5", g.Command(vy=1.5, freq=4.0)),
        g.Dodge("sustained vy 2.5 (falls)", g.Command(vy=2.5, freq=4.0)),
        replace(g.RECOMMENDED, name="hand-tuned: kick 2.0/0.3s, 1.0, 0.5s"),
        *optimized_dodges(),
        *([safe] if safe is not None else []),
    ]


def grid(n: int) -> tuple[int, int]:
    cols = 1 if n <= 3 else 2
    return -(-n // cols), cols


def layout(n: int) -> list[tuple[float, float]]:
    rows, _ = grid(n)
    return [(((rows - 1) / 2 - (i % rows)) * ROW, (i // rows) * COL) for i in range(n)]


LOWPOLY_DIR = g.SCENE.parent / "assets_lowpoly"
LOWPOLY_KEEP = 0.08            # keep ~8 % of the triangles (the Go2 visual meshes have ~200k per dog)


def ensure_lowpoly() -> bool:
    """Pre-generate decimated copies of the Go2 visual meshes (rendering only; physics is unaffected).

    The full meshes make a 6-dog frame cost ~380 ms to draw (shadow + reflection passes)."""
    src = g.SCENE.parent / "assets"
    objs = sorted(src.glob("*.obj"))
    if LOWPOLY_DIR.exists() and all((LOWPOLY_DIR / f.name).exists() for f in objs):
        return True
    try:
        import fast_simplification
        import trimesh
    except ImportError:
        print("low-poly meshes missing and trimesh/fast-simplification not installed: rendering full meshes (slow)")
        return False
    LOWPOLY_DIR.mkdir(exist_ok=True)
    before = after = 0
    for f in objs:
        mesh = trimesh.load(f, force="mesh", process=False)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)   # OBJ faces don't share vertices; the simplifier needs them to
        n = len(mesh.faces)
        keep = max(400, int(n * LOWPOLY_KEEP))
        if keep < n:
            v, fc = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=1 - keep / n)
            mesh = trimesh.Trimesh(v, fc, process=True)
        mesh.export(LOWPOLY_DIR / f.name)
        before, after = before + n, after + len(mesh.faces)
    print(f"low-poly meshes: {before} -> {after} triangles per dog, in {LOWPOLY_DIR}")
    return True


def build_world(n: int, lowpoly: bool = True):
    world = g.flat_scene_spec()
    go2_xml = g.SCENE.parent / "go2.xml"
    pos = layout(n)
    for i in range(1, n):
        child = mujoco.MjSpec.from_file(str(go2_xml))
        frame = world.worldbody.add_frame(pos=[pos[i][0], pos[i][1], 0])
        world.attach(child, frame=frame, prefix=f"d{i}_")
    world.material("groundplane").reflectance = 0.0   # the reflection pass re-draws every dog
    if lowpoly and ensure_lowpoly():
        for mesh in world.meshes:
            low = LOWPOLY_DIR / Path(mesh.file).name
            if low.exists():
                mesh.file = str(low)
    for i, (x, y0) in enumerate(pos):
        prefix = "" if i == 0 else f"d{i}_"
        world.body(prefix + "base_link").add_site(name=f"tag{i}", pos=[0, 0, 0.16], size=[0.05, 0, 0],
                                                   rgba=COLORS[i % len(COLORS)], group=0)
        # floor: lane colour patch, start line (white), 30 cm line (red)
        world.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x - 0.5, y0 + 0.15, 0.001], size=[0.06, 0.15, 0.001],
                                 rgba=COLORS[i % len(COLORS)], contype=0, conaffinity=0)
        for name, dy, rgba, w in ((f"start{i}", 0.0, (1, 1, 1, 0.9), 0.007), (f"line30_{i}", 0.30, LINE_RED, 0.009)):
            world.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y0 + dy, 0.001],
                                     size=[0.5, w, 0.001], rgba=rgba, contype=0, conaffinity=0)
    return world, pos


@dataclass
class Recording:
    t: np.ndarray                  # time since the dodge command, one entry per sim step
    qpos: np.ndarray               # [frames, nq]
    t30: list[float | None]        # per dog: time to 30 cm
    fall_t: list[float | None]     # per dog: time the fall was detected

    @property
    def start(self) -> float:
        return -LEAD_IN

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def frame(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))


class Show:
    def __init__(self, dodges: list[g.Dodge], phys: g.Physics = g.Physics()):
        torch.set_num_threads(1)
        self.dodges = dodges
        spec, pos = build_world(len(dodges))
        self.m = spec.compile()
        self.m.vis.quality.shadowsize = 2048                  # 4096 default; half the shadow-map cost
        self.m.opt.timestep = phys.timestep
        self.d = mujoco.MjData(self.m)
        self.substeps = round(g.POLICY_DT / phys.timestep)
        self.robots = [g.Go2Robot(self.m, self.d, SAFE_PHYS if dg.name == SAFE_NAME else phys,
                                  "" if i == 0 else f"d{i}_", home_xy=pos[i])
                       for i, dg in enumerate(dodges)]
        self.ctls = [g.DodgeController(dg, r, g.WTWPolicy()) for dg, r in zip(dodges, self.robots)]
        self.t0 = dodges[0].t0
        assert all(abs(dg.t0 - self.t0) < 1e-9 for dg in dodges), "all dodges must trigger at the same time"
        self.total = max(dg.total for dg in dodges)

    def simulate(self) -> Recording:
        mujoco.mj_resetData(self.m, self.d)
        for r, c in zip(self.robots, self.ctls):
            r.reset()
            c.reset()
        mujoco.mj_forward(self.m, self.d)
        n = len(self.robots)
        t, ts, qs = 0.0, [], []
        t30, fall_t = [None] * n, [None] * n
        dt = self.m.opt.timestep
        while t < self.total - 1e-9:
            tg = [c.targets(t) for c in self.ctls]
            for k in range(self.substeps):
                for r, (q, kp, kd, _) in zip(self.robots, tg):
                    r.apply_pd(q, kp, kd)
                mujoco.mj_step(self.m, self.d)
                tk = t + (k + 1) * dt - self.t0
                if tk >= -LEAD_IN - 1e-9:
                    ts.append(tk)
                    qs.append(self.d.qpos.copy())
            t += g.POLICY_DT
            for i, (c, (_, _, _, phase)) in enumerate(zip(self.ctls, tg)):
                c.record(t, phase)
                if t30[i] is None and c.log["t"][-1] >= 0 and c.log["y"][-1] >= 0.30:
                    t30[i] = c.log["t"][-1]
                if fall_t[i] is None and c.fell:
                    fall_t[i] = t - self.t0
        return Recording(np.array(ts), np.array(qs), t30, fall_t)


def overlay(dodges, rec: Recording, t: float, speed: float, playing: bool) -> tuple[str, str]:
    state = "playing" if playing else "paused"
    left = [f"t = {t:+.3f} s   (dodge command at 0)", f"{speed:.2f}x speed, {state}", ""]
    right = ["", "", ""]
    for i, dg in enumerate(dodges):
        left.append(f"{COLOR_NAMES[i % len(COLOR_NAMES)]:>6s}  {dg.name}")
        if rec.fall_t[i] is not None and t >= rec.fall_t[i]:
            right.append("FELL")
        elif rec.t30[i] is not None and t >= rec.t30[i]:
            right.append(f"30 cm in {rec.t30[i]:.2f} s")
        else:
            right.append("...")
    left += ["", "space play/pause  <- -> step  up/down speed  R restart  F fit"]
    return "\n".join(left), "\n".join(right)


def fit_camera(m: mujoco.MjModel, cam, n: int, aspect: float) -> None:
    """Perspective camera from the front (+x), looking down 50 deg, framing every lane."""
    rows, cols = grid(n)
    elev = 50.0
    x_ext = (rows - 1) * ROW + 1.2                   # depth of the grid
    y_lo, y_hi = -0.5, (cols - 1) * COL + 1.4        # start line .. where the runaway dog ends
    vert = x_ext * math.sin(math.radians(elev)) + 1.0
    horiz = y_hi - y_lo
    tan_half = math.tan(math.radians(m.vis.global_.fovy) / 2)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.15, (y_lo + y_hi) / 2, 0.15]   # nudge towards the front row (perspective makes it larger)
    cam.distance = 1.2 * max(vert / 2 / tan_half, horiz / 2 / (tan_half * max(aspect, 0.3)))
    cam.azimuth = 180.0                               # +y (dodge direction) is screen-right
    cam.elevation = -elev


def snapshot(show: Show, rec: Recording, path: str, at: float) -> None:
    show.d.qpos[:] = rec.qpos[rec.frame(at)]
    mujoco.mj_forward(show.m, show.d)
    show.m.vis.global_.offwidth, show.m.vis.global_.offheight = 1280, 720
    r = mujoco.Renderer(show.m, 720, 1280)
    cam = mujoco.MjvCamera()
    fit_camera(show.m, cam, len(show.robots), 1280 / 720)
    r.update_scene(show.d, cam)
    from PIL import Image
    Image.fromarray(r.render()).save(path)
    print(f"snapshot at t={at:+.2f}s -> {path}")
    print(overlay(show.dodges, rec, at, 1.0, False)[0])


class Player:
    """MuJoCo passive viewer for the scene + a Tk control panel driving the playback clock."""

    KEY_SPACE, KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_F, KEY_R = 32, 262, 263, 264, 265, 70, 82

    def __init__(self, show: Show, rec: Recording, speed: float, bench: float = 0.0):
        import tkinter as tk
        from tkinter import ttk
        import mujoco.viewer

        self.show, self.rec = show, rec
        self.t, self.speed, self.playing = rec.start, speed, True
        self.keys: queue.Queue[int] = queue.Queue()
        self.view_size = None
        self.v = mujoco.viewer.launch_passive(show.m, show.d, show_left_ui=False, show_right_ui=False,
                                              key_callback=self.keys.put)

        self.root = root = tk.Tk()
        root.title("Go2 dodge playback")
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        pad = dict(padx=6, pady=3)

        top = ttk.Frame(root)
        top.pack(fill="x", **pad)
        self.time_lbl = ttk.Label(top, text="", font=("Consolas", 12, "bold"), width=44)
        self.time_lbl.pack(side="left")
        for text, cmd in (("⏮ Restart", self.restart), ("◀ 10 ms", lambda: self.step(-0.01)),
                          ("⏯ Play/Pause", self.toggle), ("10 ms ▶", lambda: self.step(0.01)),
                          ("Fit view", self.fit)):
            ttk.Button(top, text=text, command=cmd).pack(side="left", padx=2)
        self.loop = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Loop", variable=self.loop).pack(side="left", padx=6)

        ttk.Label(root, text="Time since dodge command [s]").pack(anchor="w", **pad)
        self.time_var = tk.DoubleVar(value=self.t)
        self.time_scale = tk.Scale(root, from_=rec.start, to=rec.end, resolution=0.002, orient="horizontal",
                                   length=760, variable=self.time_var, command=self.on_scrub, showvalue=False,
                                   tickinterval=0.5)
        self.time_scale.pack(fill="x", **pad)

        sp = ttk.Frame(root)
        sp.pack(fill="x", **pad)
        ttk.Label(sp, text="Speed").pack(side="left")
        self.speed_var = tk.DoubleVar(value=speed)
        tk.Scale(sp, from_=0.05, to=2.0, resolution=0.05, orient="horizontal", length=420,
                 variable=self.speed_var, command=lambda v: setattr(self, "speed", float(v))).pack(side="left", padx=6)
        for s in (0.1, 0.25, 0.5, 1.0):
            ttk.Button(sp, text=f"{s:g}x", width=5, command=lambda s=s: self.set_speed(s)).pack(side="left", padx=1)

        res = ttk.Frame(root)
        res.pack(fill="x", **pad)
        ttk.Label(res, text="now", foreground="#555").grid(row=0, column=2, sticky="w", padx=12)
        ttk.Label(res, text="final", foreground="#555").grid(row=0, column=3, sticky="w", padx=12)
        self.live = []
        for i, dg in enumerate(show.dodges):
            c = COLORS[i % len(COLORS)]
            hexc = "#%02x%02x%02x" % tuple(int(255 * v) for v in c[:3])
            outcome = ("FELL" if rec.fall_t[i] is not None
                       else f"30 cm in {rec.t30[i]:.2f} s" if rec.t30[i] is not None else "never 30 cm")
            tk.Label(res, text="●", fg=hexc, font=("Segoe UI", 14)).grid(row=i + 1, column=0)
            ttk.Label(res, text=dg.name).grid(row=i + 1, column=1, sticky="w")
            lbl = ttk.Label(res, text="...", font=("Consolas", 10, "bold"), width=18)
            lbl.grid(row=i + 1, column=2, sticky="w", padx=12)
            self.live.append(lbl)
            ttk.Label(res, text=outcome, font=("Consolas", 10)).grid(row=i + 1, column=3, sticky="w", padx=12)

        m = show.m
        self.line30 = [m.geom(f"line30_{i}").id for i in range(len(show.dodges))]
        self.tags = [m.site(f"tag{i}").id for i in range(len(show.dodges))]
        self.tag_rgba = [m.site_rgba[k].copy() for k in self.tags]
        self.status = [None] * len(show.dodges)
        legend = "\n".join(f"{COLOR_NAMES[i % len(COLOR_NAMES)]:>6s}  {dg.name}" for i, dg in enumerate(show.dodges))
        legend += "\n\n30 cm line turns green when crossed; ball turns black on a fall"
        self.v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, legend, "")])
        ttk.Label(root, text="3D window: space play/pause · ←/→ step · ↑/↓ speed · R restart · F fit view · "
                             "mouse: left-drag rotate, right-drag pan, scroll zoom",
                  foreground="#555").pack(anchor="w", **pad)

        self.bench, self.bench_frames = bench, 0
        self.last = self.fps_t0 = self.start_wall = time.perf_counter()
        self.frames, self.fps = 0, 0.0
        self._setting_slider = False
        self.fit()
        root.after(2, self.tick)
        root.mainloop()

    # ---- controls ----------------------------------------------------------------------------
    def toggle(self):
        if not self.playing and self.t >= self.rec.end - 1e-9:
            self.t = self.rec.start
        self.playing = not self.playing

    def restart(self):
        self.t, self.playing = self.rec.start, True

    def step(self, dt):
        self.playing = False
        self.t = float(np.clip(self.t + dt, self.rec.start, self.rec.end))

    def set_speed(self, s):
        self.speed = float(np.clip(s, 0.05, 2.0))
        self.speed_var.set(self.speed)

    def on_scrub(self, value):
        if not self._setting_slider:
            self.t = float(value)

    def fit(self):
        vp = getattr(self.v, "viewport", None)
        aspect = vp.width / vp.height if vp is not None and vp.height > 0 else 16 / 9
        self.view_size = (vp.width, vp.height) if vp is not None else None
        with self.v.lock():
            fit_camera(self.show.m, self.v.cam, len(self.show.robots), aspect)

    def quit(self):
        try:
            self.v.close()
        finally:
            self.root.destroy()

    # ---- playback loop -----------------------------------------------------------------------
    def tick(self):
        if not self.v.is_running():
            self.root.destroy()
            return
        while not self.keys.empty():
            k = self.keys.get()
            if k == self.KEY_SPACE:
                self.toggle()
            elif k == self.KEY_RIGHT:
                self.step(0.01)
            elif k == self.KEY_LEFT:
                self.step(-0.01)
            elif k == self.KEY_UP:
                self.set_speed(self.speed * 1.5)
            elif k == self.KEY_DOWN:
                self.set_speed(self.speed / 1.5)
            elif k == self.KEY_R:
                self.restart()
            elif k == self.KEY_F:
                self.fit()

        now = time.perf_counter()
        if self.playing:
            self.t += (now - self.last) * self.speed
            if self.t > self.rec.end:
                self.t = self.rec.start if self.loop.get() else self.rec.end
                self.playing = self.loop.get()
        self.last = now

        vp = getattr(self.v, "viewport", None)            # refit when the 3D window is resized
        if vp is not None and vp.height > 0 and self.view_size != (vp.width, vp.height):
            self.fit()

        status = []
        for i in range(len(self.show.dodges)):
            if self.rec.fall_t[i] is not None and self.t >= self.rec.fall_t[i]:
                status.append("FELL")
            elif self.rec.t30[i] is not None and self.t >= self.rec.t30[i]:
                status.append(f"30 cm in {self.rec.t30[i]:.2f} s")
            else:
                status.append("...")
        with self.v.lock():
            self.show.d.qpos[:] = self.rec.qpos[self.rec.frame(self.t)]
            mujoco.mj_forward(self.show.m, self.show.d)
            for i, st in enumerate(status):
                if st != self.status[i]:
                    self.show.m.geom_rgba[self.line30[i]] = LINE_GREEN if st.startswith("30") else LINE_RED
                    self.show.m.site_rgba[self.tags[i]] = FELL_RGBA if st == "FELL" else self.tag_rgba[i]
        self.v.sync()
        for i, st in enumerate(status):
            if st != self.status[i]:
                self.live[i].config(text=st)
        self.status = status

        self._setting_slider = True
        self.time_var.set(self.t)
        self._setting_slider = False
        self.frames += 1
        self.bench_frames += 1
        if self.bench and now - self.start_wall >= self.bench:
            print(f"average {self.bench_frames / (now - self.start_wall):.1f} fps over {now - self.start_wall:.1f} s", flush=True)
            self.quit()
            return
        if now - self.fps_t0 >= 0.5:
            self.fps = self.frames / (now - self.fps_t0)
            self.frames, self.fps_t0 = 0, now
        state = "playing" if self.playing else "paused"
        self.time_lbl.config(text=f"t = {self.t:+.3f} s   {self.speed:.2f}x {state}   {self.fps:4.0f} fps")
        self.root.after(2, self.tick)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=0.5, help="initial playback speed (1 = real time)")
    ap.add_argument("--snapshot", default="", help="render one frame to this png instead of opening a window")
    ap.add_argument("--at", type=float, default=0.5, help="snapshot time after the dodge command [s]")
    ap.add_argument("--bench", type=float, default=0.0, help="play for N seconds, print the average fps, quit")
    args = ap.parse_args()

    show = Show(lineup())
    tic = time.perf_counter()
    print(f"simulating {len(show.dodges)} dodges ...", flush=True)
    rec = show.simulate()
    print(f"recorded {len(rec.t)} frames ({rec.end - rec.start:.1f} s) in {time.perf_counter() - tic:.1f} s", flush=True)
    if args.snapshot:
        snapshot(show, rec, args.snapshot, args.at)
        return
    Player(show, rec, args.speed, args.bench)


if __name__ == "__main__":
    main()
