"""Unit tests of guard.py against MuJoCo ground truth (no ROS needed).

  cd sim/avoid && python test_guard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import go2_dodge_sim as g  # noqa: E402
import guard  # noqa: E402
from lidar_sim import SimLidar  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def world(obstacles=()):
    spec = g.flat_scene_spec()
    for kind, pos, size in obstacles:
        spec.worldbody.add_geom(type=kind, pos=pos, size=size, rgba=(0.8, 0.3, 0.2, 1))
    m = spec.compile()
    d = mujoco.MjData(m)
    return m, d


def test_kinematics():
    print("leg kinematics and body height vs MuJoCo")
    m, d = world()
    r = g.Go2Robot(m, d, g.Physics())
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        q = g.DEFAULT_Q + rng.uniform(-0.4, 0.4, 12)
        q = np.clip(q, r.q_lo + 0.01, r.q_hi - 0.01)
        d.qpos[r.qadr] = q
        d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        pts = guard.leg_points(q)
        for leg in guard.LEGS:
            true_foot = d.xpos[m.body(f"{leg}_foot").id] - d.xpos[r.trunk]
            worst = max(worst, np.linalg.norm(pts[leg][3] - true_foot))
    check("foot positions match MuJoCo", worst < 1e-3, f"worst error {worst * 1000:.2f} mm")

    # standing / trotting: body height from kinematics vs truth
    sim = g.Go2Sim(g.Physics())
    pol = g.WTWPolicy()
    ctl = g.DodgeController(g.Dodge("trot", g.Command(vy=0.0)), sim, pol)
    errs = []
    while sim.t < 3.0:
        qt, kp, kd, ph = ctl.targets(sim.t)
        sim.step(qt, kp, kd)
        ctl.record(sim.t, ph)
        if sim.t > 1.0:
            h_est = guard.body_height(sim.q, guard.roll_pitch_matrix(sim.d.qpos[3:7]))
            errs.append(h_est - sim.d.qpos[2])
    errs = np.abs(errs)
    check("body height while trotting", np.percentile(errs, 95) < 0.02,
          f"p95 error {np.percentile(errs, 95) * 100:.1f} cm, max {errs.max() * 100:.1f} cm")


def run_trot_with_lidar(obstacles=(), vy=0.0, seconds=3.0, seed=1):
    """Yields (robot, points in base_footprint, obstacle pts, floor pts, q, quat at capture) per lidar frame."""
    m, d = world(obstacles)
    phys = g.Physics()
    m.opt.timestep = phys.timestep
    r = g.Go2Robot(m, d, phys)
    mujoco.mj_resetData(m, d)
    r.reset()
    mujoco.mj_forward(m, d)
    lidar = SimLidar(m, d, r.trunk, seed=seed)
    tf = guard.LidarToFootprint()
    hist = guard.StateHistory()
    ctl = g.DodgeController(g.Dodge("t", g.Command(vy=vy), pre=0.5, dodge=seconds), r, g.WTWPolicy())
    t = 0.0
    while t < 0.5 + 0.5 + seconds:
        qt, kp, kd, ph = ctl.targets(t)
        for _ in range(10):
            r.apply_pd(qt, kp, kd)
            mujoco.mj_step(m, d)
            t += m.opt.timestep
            hist.add(t, d.qpos[3:7], r.q)
            for stamp, pts in lidar.update(t):
                state = hist.at(stamp)
                if t > 1.0 and state is not None:
                    quat, q = state
                    p_fp = tf(pts, quat, q)
                    obs, floor = guard.self_and_floor(p_fp, q, quat)
                    yield r, p_fp, obs, floor, q, quat
        ctl.record(t, ph)


def test_lidar_frame_and_self_filter():
    print("lidar -> base_footprint, floor height, self filter (open floor)")
    floor_z, leaks, frames, pts_n = [], 0, 0, []
    for vy in (0.0, 1.0):          # trotting in place and fast sideways walking
        for r, p_fp, obs, floor, *_ in run_trot_with_lidar(vy=vy, seconds=2.0):
            frames += 1
            pts_n.append(len(p_fp))
            floor_z.extend(floor[:, 2])
            leaks += len(obs)
    floor_z = np.array(floor_z)
    check("points per frame ~ L1 rate", 800 < np.mean(pts_n) < 2000, f"mean {np.mean(pts_n):.0f}")
    check("floor lands at z~0 in base_footprint", abs(np.median(floor_z)) < 0.02,
          f"median {np.median(floor_z) * 100:.1f} cm, p5..p95 {np.percentile(floor_z, 5) * 100:.1f}.."
          f"{np.percentile(floor_z, 95) * 100:.1f} cm")
    check("no self points leak as obstacles (open floor)", leaks <= frames * 2,
          f"{leaks} leaked points in {frames} frames")


def test_side_room():
    print("side room with a wall at known distance")
    for dist in (0.15, 0.35, 0.6, 0.9):
        y_wall = 0.20 + dist                       # body half width 0.20 + free room
        wall = (mujoco.mjtGeom.mjGEOM_BOX, [0, y_wall + 0.05, 0.25], [1.0, 0.05, 0.25])
        rooms, seen_r = [], []
        recent = guard.RecentFrames(3)            # what the controller decides on: the last 3 frames
        for r, p_fp, obs, floor, *_ in run_trot_with_lidar([wall], seconds=1.5, seed=int(dist * 100)):
            recent.add(obs, floor)
            # true room: wall face minus body half width, from the trunk's lateral position
            true = y_wall - r.d.qpos[1] - 0.20
            room = guard.side_room(obs, floor, +1, need=0.0)[0]           # single frame: distance
            rooms.append(room - true)
            if len(recent.frames) == 3:
                seen_r.append(guard.side_room(*recent.merged(), -1, need=0.8)[1])
        rooms = np.array(rooms)
        # over-estimating room is the dangerous direction: at most +2 cm; under-estimates up to 6 cm are fine
        check(f"wall at {dist:.2f} m: room measured (+2/-6 cm)", bool(np.all((rooms < 0.02) & (rooms > -0.06))),
              f"error {rooms.mean() * 100:+.1f} cm, range {rooms.min() * 100:+.1f}..{rooms.max() * 100:+.1f} cm")
        # availability, not safety: how often the open side is confirmed free (stricter floor check costs some)
        check(f"wall at {dist:.2f} m: open right side confirmed free in >= 60% of 3-frame windows",
              bool(np.mean(seen_r) >= 0.6), f"{100 * np.mean(seen_r):.0f}%")


def test_choose_side_and_watchdog():
    print("choose_side and watchdog logic")
    need = 0.8
    check("both free -> preferred", guard.choose_side((1.0, True), (1.0, True), need) == 1)
    check("left blocked -> right", guard.choose_side((0.3, True), (1.0, True), need) == -1)
    check("left floor unseen -> right", guard.choose_side((9, False), (1.0, True), need) == -1)
    check("both blocked -> 0", guard.choose_side((0.3, True), (0.5, True), need) == 0)
    check("both unseen -> 0", guard.choose_side((9, False), (9, False), need) == 0)
    def wall_pts(y, n=10):
        return np.array([[x, y, 0.2] for x in np.linspace(-0.3, 0.3, n)])
    check("keep_off: dodged left, obstacle 5 cm left -> move right", guard.keep_off(wall_pts(0.25), +1) < 0)
    check("keep_off: dodged right, obstacle 5 cm right -> move left", guard.keep_off(wall_pts(-0.25), -1) > 0)
    check("keep_off: never moves towards the dodge side", guard.keep_off(wall_pts(-0.25), +1) == 0)
    check("keep_off: other side also near -> stay",
          guard.keep_off(np.vstack([wall_pts(0.25), wall_pts(-0.30)]), +1) == 0)
    check("keep_off: obstacle 30 cm away -> stay", guard.keep_off(wall_pts(0.50), +1) == 0)
    check("keep_off: 2 stray points ignored", guard.keep_off(wall_pts(0.22, n=2), +1) == 0)
    check("keep_off: no dodge -> stay", guard.keep_off(wall_pts(0.25), 0) == 0)
    rf = guard.RecentFrames(3)
    for k in range(5):
        rf.add(np.full((k + 1, 3), k), np.zeros((1, 3)))
    o, f = rf.merged()
    check("RecentFrames keeps the last 3 frames", len(o) == 3 + 4 + 5 and len(f) == 3 and o.min() == 2)
    w = guard.Watchdog(max_cloud_age=0.3, max_monitor_age=0.06)
    check("no data -> zero", not w.command(1.0).any())
    w.cloud(1.0)
    w.monitor(1.0, [0, 1.2, 0])
    check("fresh -> passes", w.command(1.02)[1] == 1.2)
    check("monitor silent 0.1 s -> zero", not w.command(1.1).any())
    w.monitor(1.4, [0, 1.2, 0])
    check("lidar 0.4 s old -> zero", not w.command(1.4).any())


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_choose_side_and_watchdog()
    test_kinematics()
    test_lidar_frame_and_self_filter()
    test_side_room()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
