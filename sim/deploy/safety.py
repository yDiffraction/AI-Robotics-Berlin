"""Safety rules for running our own low-level controller on the Go2 (sport mode released).

Pure logic, no DDS, no MuJoCo: the supervisor process (supervisor.py) feeds it messages and asks
check() every control tick; the tests drive it directly.

Once sport mode is released, Unitree's own protections are gone: the remote's L2+B damping, fall
handling, and whatever happens when nobody sends commands. Everything here trips into one latched
state, DAMPING (kp=0, kd=5 on every joint, the same command walk-these-ways' deploy code and the
Unitree examples use), and stays there until the process ends.

Triggers
  e-stop        L2+B on the remote, read from rt/lowstate.wireless_remote AND rt/wirelesscontroller
  operator      Ctrl+C / SIGTERM to the supervisor
  tilt          |roll| or |pitch| > 0.8 rad (walk-these-ways' threshold; the SAFE dodge peaks ~0.35)
  controller    no valid rt/lowcmd from our controller for 50 ms (crash, hang, wrong CRC), or a
                command that is NaN/inf, outside the joint range, or has absurd gains
  robot link    no valid rt/lowstate for 100 ms, or its tick counter stops for 100 ms
  remote link   rt/wirelesscontroller silent for 0.5 s (only after it has been seen; see README:
                whether the Go2 stops publishing when the remote is off must be checked once)
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

# ---- Go2 conventions (unitree_sdk2 order: FR, FL, RR, RL; hip, thigh, calf) --------------------
SDK_LEGS = ("FR", "FL", "RR", "RL")
SDK_JOINTS = [f"{leg}_{p}_joint" for leg in SDK_LEGS for p in ("hip", "thigh", "calf")]
# joint ranges from unitree_mujoco go2.xml (same as the URDF)
_HIP, _FRONT_THIGH, _REAR_THIGH, _CALF = (-1.0472, 1.0472), (-1.5708, 3.4907), (-0.5236, 4.5379), (-2.7227, -0.83776)
Q_RANGE = np.array([_HIP, _FRONT_THIGH, _CALF, _HIP, _FRONT_THIGH, _CALF,
                    _HIP, _REAR_THIGH, _CALF, _HIP, _REAR_THIGH, _CALF])

# remote buttons: bit index in the uint16 at wireless_remote[2:4] (= WirelessController_.keys)
# (unitree_sdk2_python example/wireless_controller, walk-these-ways-go2 xKeySwitchUnion)
BUTTON_BIT = {"R1": 0, "L1": 1, "Start": 2, "Select": 3, "R2": 4, "L2": 5, "F1": 6, "F2": 7,
              "A": 8, "B": 9, "X": 10, "Y": 11, "Up": 12, "Right": 13, "Down": 14, "Left": 15}
ESTOP_KEYS = (1 << BUTTON_BIT["L2"]) | (1 << BUTTON_BIT["B"])

DAMP_KD = 5.0


def keys_from_remote_bytes(raw) -> int:
    """Button bit field from LowState_.wireless_remote (40 bytes)."""
    return struct.unpack_from("<H", bytes(raw), 2)[0]


def remote_bytes(keys: int) -> list[int]:
    """Inverse, for tests and the fake robot: 40 bytes with only the buttons set."""
    raw = [0] * 40
    raw[2], raw[3] = keys & 0xFF, keys >> 8
    return raw


def is_estop(keys: int) -> bool:
    return (keys & ESTOP_KEYS) == ESTOP_KEYS


def check_command(q, dq, kp, kd, tau, mode) -> str | None:
    """Reason why a LowCmd for the 12 leg motors must not reach the robot, or None.
    Arrays in SDK order. Generous bounds: they catch garbage, not tuning."""
    a = np.array([q, dq, kp, kd, tau], dtype=float)
    if not np.isfinite(a).all():
        return "NaN/inf in command"
    if (np.asarray(mode) != 0x01).any():
        return "motor mode not 0x01"
    if (a[2] < 0).any() or (a[2] > 80).any():
        return f"kp outside 0..80 (max {a[2].max():.1f})"
    if (a[3] < 0).any() or (a[3] > 8).any():
        return f"kd outside 0..8 (max {a[3].max():.1f})"
    if (np.abs(a[4]) > 10).any():
        return f"feed-forward torque > 10 Nm ({np.abs(a[4]).max():.1f})"
    if (np.abs(a[1]) > 40).any():
        return "velocity target > 40 rad/s"
    held = a[2] > 0                         # position targets only matter where kp > 0
    lo, hi = Q_RANGE[:, 0] - 0.05, Q_RANGE[:, 1] + 0.05
    bad = held & ((a[0] < lo) | (a[0] > hi))
    if bad.any():
        j = int(np.nonzero(bad)[0][0])
        return f"target outside joint range: {SDK_JOINTS[j]} = {a[0][j]:.2f}"
    return None


def is_lying(q_sdk) -> bool:
    """All four legs folded (thigh > 0.9, calf < -2.0 rad; standing is about 0.8 / -1.5): the trunk is on
    or just above the floor. Unitree's own lie-down pose is thigh 1.36, calf -2.65."""
    q = np.asarray(q_sdk, dtype=float).reshape(4, 3)
    return bool((q[:, 1] > 0.9).all() and (q[:, 2] < -2.0).all())


def tilt_from_quat(wxyz) -> tuple[float, float]:
    """Roll, pitch [rad] from the IMU quaternion (w, x, y, z), as in LowState_.imu_state.quaternion."""
    w, x, y, z = wxyz
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


@dataclass
class Limits:
    cmd_timeout: float = 0.05       # controller silent (robot keeps stepping on stale targets meanwhile)
    state_timeout: float = 0.10     # rt/lowstate silent (published at ~500 Hz)
    tick_timeout: float = 0.10      # rt/lowstate arrives but its tick counter is stuck
    remote_timeout: float = 0.5     # rt/wirelesscontroller silent
    tilt: float = 0.8               # rad, walk-these-ways deploy threshold
    startup_timeout: float = 60.0   # controller must send its first command within this
    remote_check: bool = True       # False: do not trip on a silent rt/wirelesscontroller


class Supervisor:
    """Latched trip logic. Times are seconds on one monotonic clock (the caller's)."""

    def __init__(self, limits: Limits | None = None, now: float = 0.0):
        self.lim = limits or Limits()
        self.t_start = now
        self.trip_reason: str | None = None
        self.trip_t: float | None = None
        self.last_state = self.last_cmd = self.last_remote = None
        self.last_tick = None
        self.tick_changed_t = None
        self.tick_seen_moving = False
        self.controller_started = False     # armed: the controller has sent a valid command
        self.events: list[tuple[float, str]] = []

    @property
    def tripped(self) -> bool:
        return self.trip_reason is not None

    def trip(self, now: float, reason: str) -> None:
        if self.trip_reason is None:
            self.trip_reason, self.trip_t = reason, now
            self.events.append((now, "TRIP: " + reason))

    # ---- inputs ----------------------------------------------------------------------------
    def on_state(self, now: float, crc_ok: bool, tick: int, quat_wxyz, remote_raw) -> None:
        if not crc_ok:
            return                            # a corrupted message counts as a missing one
        self.last_state = now
        if tick != self.last_tick:
            if self.last_tick is not None:
                self.tick_seen_moving = True
            self.last_tick, self.tick_changed_t = tick, now
        if is_estop(keys_from_remote_bytes(remote_raw)):
            self.trip(now, "e-stop (L2+B, lowstate)")
        roll, pitch = tilt_from_quat(quat_wxyz)
        if self.controller_started and max(abs(roll), abs(pitch)) > self.lim.tilt:
            self.trip(now, f"tilt roll {roll:+.2f} pitch {pitch:+.2f} rad")

    def on_remote(self, now: float, keys: int) -> None:
        self.last_remote = now
        if is_estop(keys):
            self.trip(now, "e-stop (L2+B, wirelesscontroller)")

    def on_command(self, now: float, crc_ok: bool, problem: str | None) -> None:
        """A LowCmd from our controller (not from the supervisor itself)."""
        if not crc_ok:
            return                            # the robot drops it too: counts as missing
        if problem:
            self.trip(now, "controller sent a bad command: " + problem)
            return
        self.last_cmd = now
        self.controller_started = True

    def on_operator_stop(self, now: float, what: str = "Ctrl+C") -> None:
        self.trip(now, f"operator ({what})")

    # ---- periodic check --------------------------------------------------------------------
    def check(self, now: float) -> str | None:
        """Call every control tick. Returns the trip reason once tripped (latched)."""
        if self.tripped:
            return self.trip_reason
        L = self.lim
        if not self.controller_started:
            if now - self.t_start > L.startup_timeout:
                self.trip(now, f"controller sent no valid command within {L.startup_timeout:.0f} s")
            return self.trip_reason             # not armed yet: nothing is being sent to the robot
        if now - self.last_cmd > L.cmd_timeout:
            self.trip(now, f"controller silent for {(now - self.last_cmd) * 1000:.0f} ms")
        elif self.last_state is None or now - self.last_state > L.state_timeout:
            age = "never" if self.last_state is None else f"{(now - self.last_state) * 1000:.0f} ms"
            self.trip(now, f"robot state (rt/lowstate) lost: {age}")
        elif self.tick_seen_moving and now - self.tick_changed_t > L.tick_timeout:
            self.trip(now, f"robot tick stuck for {(now - self.tick_changed_t) * 1000:.0f} ms")
        elif L.remote_check and self.last_remote is not None and now - self.last_remote > L.remote_timeout:
            self.trip(now, f"remote link (rt/wirelesscontroller) silent for {now - self.last_remote:.2f} s")
        return self.trip_reason
