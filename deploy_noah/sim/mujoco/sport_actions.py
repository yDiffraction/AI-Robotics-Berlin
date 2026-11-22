"""Canned joint trajectories standing in for the Go2 sport-mode actions the starter's action_bridge sends.

These are hand-tuned PD keyframes for the MuJoCo sim, not Unitree's controller: good enough to see which
action fired and whether the robot ends up where the behaviour logic expects, not for judging real motion.

Joint order (as in rt/lowcmd): FR, FL, RR, RL, each hip / thigh / calf.
"""
import math

import numpy as np

KP, KD = 60.0, 2.0  # kd above ~4 chatters at the 200 Hz sim step (robot skates across the floor)


def legs(fr, fl, rr, rl):
    return np.array([*fr, *fl, *rr, *rl], dtype=float)


def mirrored(front, rear):
    """Same pose left and right; hip abduction is mirrored for the right legs."""
    fh, ft, fc = front
    rh, rt, rc = rear
    return legs((-fh, ft, fc), (fh, ft, fc), (-rh, rt, rc), (rh, rt, rc))


STAND = mirrored((0.0, 0.67, -1.3), (0.0, 0.67, -1.3))
LAY = mirrored((0.05, 1.22, -2.44), (0.05, 1.22, -2.44))
SIT = mirrored((0.0, 1.15, -0.9), (0.0, 1.95, -2.5))
BOW = mirrored((0.0, 1.25, -2.3), (0.0, 0.55, -1.1))


def with_leg(pose, leg, joints):
    out = pose.copy()
    out[3 * leg:3 * leg + 3] = joints
    return out


def _hello():
    # Wave the front-right paw from the sit pose, leaning onto the left side first.
    lean = SIT.copy()
    lean[[0, 3, 6, 9]] = 0.32
    up = with_leg(lean, 0, (0.32, -0.5, -1.5))
    out = with_leg(lean, 0, (0.32, -0.9, -0.9))
    frames = [(1.2, SIT), (0.4, lean), (0.4, up)]
    for _ in range(3):
        frames += [(0.25, out), (0.25, up)]
    return frames + [(0.4, lean), (0.3, SIT), (0.8, STAND)]


def _wiggle(cycles, period, sway=0.22, bob=0.0):
    frames = []
    steps = 8
    for k in range(1, cycles * steps + 1):
        phase = 2 * math.pi * k / steps
        pose = STAND.copy()
        pose[[0, 3, 6, 9]] += sway * math.sin(phase)
        squat = bob * (0.5 - 0.5 * math.cos(2 * phase))
        pose[[1, 4, 7, 10]] += squat
        pose[[2, 5, 8, 11]] -= 2 * squat
        frames.append((period / steps, pose))
    return frames + [(0.4, STAND)]


# api_id -> (name, keyframes as (seconds, target pose)); each segment eases from the previous target.
ACTIONS = {
    1001: ("damp", [(1.5, LAY)]),
    1002: ("balance stand", [(1.0, STAND)]),
    1004: ("stand up", [(0.6, LAY), (1.6, STAND)]),
    1005: ("stand down", [(2.0, LAY)]),
    1006: ("recovery stand", [(0.6, LAY), (1.6, STAND)]),
    1009: ("sit", [(2.0, SIT)]),
    1010: ("rise from sit", [(1.5, STAND)]),
    1016: ("hello", _hello()),
    1017: ("stretch", [(1.2, BOW), (0.8, BOW), (1.0, STAND)]),
    1020: ("content", _wiggle(cycles=3, period=0.7, sway=0.15)),
    1022: ("dance", _wiggle(cycles=8, period=1.5, sway=0.22, bob=0.25)),
    1029: ("greet (approximated by hello)", _hello()),
    1036: ("heart (approximated by hello)", _hello()),
}

STARTUP = [(1.0, LAY), (2.0, STAND)]


class Trajectory:
    """Eases through keyframes starting from wherever the previous command left the joints."""

    def __init__(self, start, frames):
        self.points = [np.asarray(start, dtype=float)] + [pose for _, pose in frames]
        self.ends = np.cumsum([seconds for seconds, _ in frames])

    @property
    def duration(self):
        return float(self.ends[-1])

    def target(self, t):
        if t >= self.ends[-1]:
            return self.points[-1]
        i = int(np.searchsorted(self.ends, t, side="right"))
        begin = self.ends[i - 1] if i else 0.0
        s = (t - begin) / (self.ends[i] - begin)
        s = s * s * (3 - 2 * s)
        return (1 - s) * self.points[i] + s * self.points[i + 1]
