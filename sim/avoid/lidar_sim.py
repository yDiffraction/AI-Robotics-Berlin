"""Simulated Unitree L1 lidar on the Go2 (MuJoCo ray casting).

Model of the real sensor (Unitree spec page: 360 x 90 deg FOV, 21600 effective points/s, 11 Hz
rotation) mounted as in the go2_description URDF (radar_joint): a hemisphere around the sensor's
+z axis, which points mostly down and slightly forward. Each frame casts a fresh random set of
rays (the L1 scan pattern is non-repetitive), adds range noise, and is delivered after a latency.

Rays hit the floor, obstacles and the robot's own legs (collision geoms, group 3) -- the real
lidar sees its legs too -- but not the trunk body, whose head collision sphere encloses the
sensor origin.
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from guard import LIDAR_XYZ, R_LIDAR


class SimLidar:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, trunk_body: int, rate: float = 11.0,
                 points_per_s: float = 21600.0, noise: float = 0.02, latency: float = 0.05,
                 max_range: float = 8.0, min_range: float = 0.05, seed: int = 0, groups=(0, 3)):
        self.m, self.d, self.trunk = m, d, trunk_body
        self.period, self.n = 1.0 / rate, int(points_per_s / rate)
        self.noise, self.latency = noise, latency
        self.max_range, self.min_range = max_range, min_range
        self.rng = np.random.default_rng(seed)
        self.geomgroup = np.zeros(6, np.uint8)
        self.geomgroup[list(groups)] = 1
        self.next_scan = 0.0
        self.queue: deque[tuple[float, float, np.ndarray]] = deque()   # (deliver_at, stamp, points)
        self.enabled = True

    def _directions(self) -> np.ndarray:
        z = self.rng.uniform(0.0, 1.0, self.n)                  # uniform over the +z hemisphere
        az = self.rng.uniform(0.0, 2 * np.pi, self.n)
        r = np.sqrt(1 - z * z)
        return np.stack([r * np.cos(az), r * np.sin(az), z], 1)

    def scan(self, t: float) -> np.ndarray:
        """One frame of points in the lidar frame (cast now)."""
        R_base = self.d.xmat[self.trunk].reshape(3, 3)
        p_lidar = self.d.xpos[self.trunk] + R_base @ LIDAR_XYZ
        R_l = R_base @ R_LIDAR
        dirs = self._directions()
        vec = (dirs @ R_l.T).astype(np.float64).ravel()
        dist = np.zeros(self.n)
        geomid = np.zeros(self.n, np.int32)
        try:        # MuJoCo >= 3.4 / 4.x: extra `normal` output argument
            mujoco.mj_multiRay(self.m, self.d, p_lidar.astype(np.float64), vec, self.geomgroup, 1, self.trunk,
                               geomid, dist, None, self.n, self.max_range)
        except TypeError:   # MuJoCo 3.3 (the WSL/ROS environment)
            mujoco.mj_multiRay(self.m, self.d, p_lidar.astype(np.float64), vec, self.geomgroup, 1, self.trunk,
                               geomid, dist, self.n, self.max_range)
        ok = (dist > self.min_range) & (dist < self.max_range) & (geomid >= 0)
        dist = dist[ok] + self.rng.normal(0, self.noise, ok.sum())
        return dirs[ok] * dist[:, None]

    def update(self, t: float) -> list[tuple[float, np.ndarray]]:
        """Call every sim step; returns frames (stamp, points) whose latency has elapsed."""
        if self.enabled and t >= self.next_scan:
            self.queue.append((t + self.latency, t, self.scan(t)))
            self.next_scan = t + self.period
        out = []
        while self.queue and self.queue[0][0] <= t:
            _, stamp, pts = self.queue.popleft()
            out.append((stamp, pts))
        return out
