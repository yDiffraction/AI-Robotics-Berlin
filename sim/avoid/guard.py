"""Obstacle guard for the low-level dodge controller. Robot-side code: numpy only, no simulator.

Unitree's own obstacle stop lives inside its sport-mode controller and is gone once we drive the
motors ourselves, so these pieces sit around the controller instead:

  LidarToFootprint   Go2 L1 lidar points (frame `utlidar_lidar`) -> `base_footprint` (on the
                     ground under the body, yaw-aligned, z up). Body roll/pitch comes from the IMU
                     and body height from leg kinematics, so no odometry is needed.
  self_and_floor     drops floor points and the robot's own legs/trunk (the lidar sees them).
  side_room          free lateral room left/right of the body, and whether the floor there was
                     actually observed (unobserved = not free).
  choose_side        dodge side with room for the whole dodge, or 0 = do not dodge.
  Watchdog           velocity passes only if Nav2 Collision Monitor answered recently AND the
                     lidar data is fresh; anything else -> zero velocity.

Nav2 Collision Monitor (proven, unmodified) does the velocity limiting in between: it receives the
filtered cloud from here and the desired velocity, and returns a velocity that will not drive the
footprint into an obstacle within its time horizon.

Frames/geometry are from Unitree's go2_description URDF and the unitree_mujoco Go2 model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# radar_joint in unitreerobotics/unitree_ros go2_description.urdf: parent `base`, xyz, rpy
LIDAR_XYZ = np.array([0.28945, 0.0, -0.046825])
LIDAR_PITCH = 2.8782

# leg geometry (go2.xml / URDF): hip joint position in base, thigh offset (y), link lengths
HIP_POS = {"FL": (0.1934, 0.0465), "FR": (0.1934, -0.0465), "RL": (-0.1934, 0.0465), "RR": (-0.1934, -0.0465)}
THIGH_Y, THIGH_LEN, CALF_LEN, FOOT_R = 0.0955, 0.213, 0.213, 0.022
LEGS = ("FL", "FR", "RL", "RR")       # joint order q[0:12] = FL hip,thigh,calf, FR ..., RL ..., RR ...


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


R_LIDAR = rot_y(LIDAR_PITCH)


def leg_points(q: np.ndarray) -> dict[str, np.ndarray]:
    """Hip, knee and foot-centre positions (base frame) of each leg from the 12 joint angles."""
    out = {}
    for i, leg in enumerate(LEGS):
        qh, qt, qc = q[3 * i:3 * i + 3]
        hx, hy = HIP_POS[leg]
        side = 1.0 if leg[1] == "L" else -1.0
        hip = np.array([hx, hy, 0.0])
        rx = rot_x(qh)
        thigh_root = hip + rx @ np.array([0.0, side * THIGH_Y, 0.0])
        knee = thigh_root + rx @ rot_y(qt) @ np.array([0.0, 0.0, -THIGH_LEN])
        foot = knee + rx @ rot_y(qt) @ rot_y(qc) @ np.array([0.0, 0.0, -CALF_LEN])
        out[leg] = np.stack([hip, thigh_root, knee, foot])
    return out


def roll_pitch_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Body orientation without yaw (IMU quaternion w,x,y,z -> R_tilt)."""
    w, x, y, z = quat_wxyz
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ r


def body_height(q: np.ndarray, r_tilt: np.ndarray) -> float:
    """Trunk origin height above the ground: mean of the two lowest feet (stance feet in a trot)."""
    feet = np.array([(r_tilt @ pts[3])[2] for pts in leg_points(q).values()])
    return float(-np.mean(np.sort(feet)[:2]) + FOOT_R)


@dataclass
class GuardConfig:
    floor_cut: float = 0.05        # points lower than this above the ground count as floor [m]
    top_cut: float = 1.0           # ignore points higher than this (the dog fits under) [m]
    leg_radius: float = 0.07       # points this close to a leg segment are the robot itself [m]
    trunk_half: tuple = (0.40, 0.17, 0.12)   # trunk box (x, y, z half sizes) around the body origin [m]
    head_front: float = 0.47       # the column under the trunk extends to the head's front [m]
    body_half_width: float = 0.20  # lateral half width incl. legs, for room measurement [m]
    corridor_x: tuple = (-0.45, 0.45)   # length of the side corridor along the body [m]
    floor_bin: float = 0.10        # floor-observation check: every bin this wide must have floor points
    min_floor_pts: int = 2
    min_obstacle_pts: int = 3      # an obstacle needs this many points (single stray returns are noise)
    floor_segments: int = 3        # floor must be seen in each front/middle/rear part of the corridor:
                                   # a box beside the hind legs hides only the rear part (measured miss)
    min_floor_pts_segment: int = 1


class LidarToFootprint:
    """Transforms lidar points into base_footprint using IMU tilt + leg-kinematic body height."""

    def __call__(self, pts_lidar: np.ndarray, quat_wxyz: np.ndarray, q: np.ndarray) -> np.ndarray:
        r_tilt = roll_pitch_matrix(quat_wxyz)
        h = body_height(q, r_tilt)
        p_base = pts_lidar @ R_LIDAR.T + LIDAR_XYZ
        p_fp = p_base @ r_tilt.T
        p_fp[:, 2] += h
        return p_fp


def _dist_to_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of every point p[N,3] to every segment a[M,3]-b[M,3] -> [N, M]."""
    ab = b - a
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - closest, axis=-1)


def self_and_floor(p_fp: np.ndarray, q: np.ndarray, quat_wxyz: np.ndarray, cfg: GuardConfig = GuardConfig()
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Split base_footprint points into (obstacle points, floor points); the robot's own body is removed."""
    r_tilt = roll_pitch_matrix(quat_wxyz)
    h = body_height(q, r_tilt)
    origin = np.array([0.0, 0.0, h])
    # robot body in base_footprint: trunk box (tilted) and leg segments
    local = (p_fp - origin) @ r_tilt          # back into the (tilted) body frame
    in_trunk = np.all(np.abs(local) <= np.array(cfg.trunk_half), axis=1)
    # nothing that matters can be under the robot's own trunk; straight-down rays put their range
    # noise fully into z, so floor points there would otherwise poke above floor_cut
    in_trunk |= (local[:, 0] >= -cfg.trunk_half[0]) & (local[:, 0] <= cfg.head_front) & \
        (np.abs(local[:, 1]) <= cfg.trunk_half[1])          # incl. the floor under the head (measured leak)
    segs_a, segs_b = [], []
    for pts in leg_points(q).values():
        w = pts @ r_tilt.T + origin
        segs_a += [w[1], w[2]]
        segs_b += [w[2], w[3]]
    near_leg = _dist_to_segments(p_fp, np.array(segs_a), np.array(segs_b)).min(1) <= cfg.leg_radius \
        if len(p_fp) else np.zeros(0, bool)
    keep = ~(in_trunk | near_leg)
    z = p_fp[:, 2]
    floor = keep & (z < cfg.floor_cut)
    obstacle = keep & (z >= cfg.floor_cut) & (z <= cfg.top_cut)
    return p_fp[obstacle], p_fp[floor]


def side_room(obstacles: np.ndarray, floor: np.ndarray, side: int, need: float,
              cfg: GuardConfig = GuardConfig()) -> tuple[float, bool]:
    """Free lateral room on `side` (+1 left, -1 right) and whether the floor over `need` was observed."""
    x0, x1 = cfg.corridor_x
    w = cfg.body_half_width

    def lateral(p):
        return side * p[:, 1] - w if len(p) else np.zeros(0)

    if len(obstacles):
        inside = (obstacles[:, 0] >= x0) & (obstacles[:, 0] <= x1) & (side * obstacles[:, 1] > 0)
        lat = np.sort(lateral(obstacles[inside]))
        k = cfg.min_obstacle_pts
        room = float(max(0.0, lat[k - 1])) if len(lat) >= k else float("inf")
    else:
        room = float("inf")
    observed = True
    if len(floor):
        f_in = floor[(floor[:, 0] >= x0) & (floor[:, 0] <= x1)]
        lat = lateral(f_in)
        seg_edges = np.linspace(x0, x1, cfg.floor_segments + 1)
        seg = np.clip(np.searchsorted(seg_edges, f_in[:, 0], side="right") - 1, 0, cfg.floor_segments - 1)
        for b0 in np.arange(0.0, need, cfg.floor_bin):
            in_bin = (lat >= b0) & (lat < b0 + cfg.floor_bin)
            if np.count_nonzero(in_bin) < cfg.min_floor_pts or \
                    np.bincount(seg[in_bin], minlength=cfg.floor_segments).min() < cfg.min_floor_pts_segment:
                observed = False
                break
    else:
        observed = False
    return room, observed


def keep_off(obstacles: np.ndarray, dodge_side: int, near: float = 0.20, speed: float = 0.30,
             other_free: float = 0.30, cfg: GuardConfig = GuardConfig()) -> float:
    """Sideways velocity that backs the robot away from an obstacle on the side it just dodged to.

    Measured in test_avoid pop-up runs: after the dodge is cut short next to an obstacle, the
    walk-these-ways policy keeps creeping ~0.1 m/s towards the dodge side for ~0.5 s with a zero
    command, and a swinging foot that reaches the obstacle trips the dog.
    Only ever moves AWAY from `dodge_side` (the only side the dodge can have made dangerous; false
    points from spread legs once pushed an unrestricted version towards the obstacle), and only if
    the opposite side has `other_free` of measured room. Returns 0 otherwise.
    Nav2 Collision Monitor zeroes ALL motion once points are inside its footprint (Humble), even
    motion away from them, so this command has to go around it (still behind the lidar watchdog)."""
    if dodge_side == 0:
        return 0.0
    here = side_room(obstacles, np.zeros((0, 3)), dodge_side, 0.0, cfg)[0]
    there = side_room(obstacles, np.zeros((0, 3)), -dodge_side, 0.0, cfg)[0]
    if here < near and there >= other_free:
        return -dodge_side * speed
    return 0.0


class RecentFrames:
    """The last few filtered frames, merged for the side decision. One L1 frame (~1800 random rays)
    leaves some 10 cm floor strip out to 0.8 m unobserved in ~3-6 % of frames (measured), which
    would veto a free side; three frames (0.27 s) observed it in 100 %. The robot trots in place
    before the trigger, so it moves only a few cm across them."""

    def __init__(self, n: int = 3):
        self.n = n
        self.frames: list[tuple[np.ndarray, np.ndarray]] = []

    def add(self, obstacles: np.ndarray, floor: np.ndarray) -> None:
        self.frames = (self.frames + [(obstacles, floor)])[-self.n:]

    def merged(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.frames:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return (np.concatenate([f[0] for f in self.frames]), np.concatenate([f[1] for f in self.frames]))


def choose_side(left: tuple[float, bool], right: tuple[float, bool], need: float, prefer: int = 1) -> int:
    """+1/-1 for the side whose room covers the whole dodge and whose floor was seen, else 0."""
    ok = {1: left[0] >= need and left[1], -1: right[0] >= need and right[1]}
    if ok[prefer]:
        return prefer
    if ok[-prefer]:
        return -prefer
    return 0


class StateHistory:
    """Recent (time, IMU quaternion, joint angles); lidar frames must be interpreted with the robot
    state at their capture time, not the current one (the trot rocks the body and swings the legs
    by ~10 cm within a typical 50 ms lidar latency)."""

    def __init__(self, horizon: float = 0.5):
        self.horizon = horizon
        self.t: list[float] = []
        self.quat: list[np.ndarray] = []
        self.q: list[np.ndarray] = []

    def add(self, t: float, quat_wxyz: np.ndarray, q: np.ndarray) -> None:
        self.t.append(t)
        self.quat.append(np.array(quat_wxyz, float))
        self.q.append(np.array(q, float))
        while self.t and self.t[0] < t - self.horizon:
            self.t.pop(0)
            self.quat.pop(0)
            self.q.pop(0)

    def at(self, stamp: float) -> tuple[np.ndarray, np.ndarray] | None:
        """State closest to `stamp`, or None if the history does not cover it (then: drop the frame)."""
        if not self.t or stamp < self.t[0] - 0.01 or stamp > self.t[-1] + 0.01:
            return None
        i = int(np.argmin(np.abs(np.array(self.t) - stamp)))
        return self.quat[i], self.q[i]


class Watchdog:
    """Pass a velocity only if the Collision Monitor answered for this cycle and the lidar is fresh."""

    def __init__(self, max_cloud_age: float = 0.30, max_monitor_age: float = 0.06):
        self.max_cloud_age, self.max_monitor_age = max_cloud_age, max_monitor_age
        self.cloud_stamp = -1e9
        self.monitor_time = -1e9
        self.monitor_cmd = np.zeros(3)
        self.reason = "startup"

    def cloud(self, stamp: float) -> None:
        self.cloud_stamp = max(self.cloud_stamp, stamp)

    def monitor(self, now: float, cmd: np.ndarray) -> None:
        self.monitor_time, self.monitor_cmd = now, np.asarray(cmd, float)

    def command_fresh_lidar(self, now: float) -> bool:
        """Lidar-only check, for the keep-off motion that cannot go through the monitor."""
        return now - self.cloud_stamp <= self.max_cloud_age

    def command(self, now: float) -> np.ndarray:
        if now - self.cloud_stamp > self.max_cloud_age:
            self.reason = f"lidar stale ({now - self.cloud_stamp:.2f} s)"
            return np.zeros(3)
        if now - self.monitor_time > self.max_monitor_age:
            self.reason = f"no collision-monitor answer ({now - self.monitor_time:.2f} s)"
            return np.zeros(3)
        self.reason = "ok"
        return self.monitor_cmd.copy()
