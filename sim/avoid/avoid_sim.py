"""MuJoCo Go2 + simulated L1 lidar + the real Nav2 Collision Monitor, in lockstep.

Per 20 ms control step:
  1. lidar frames that have arrived -> guard (robot state at capture time, self/floor filter)
     -> PointCloud2 /avoid/cloud_obstacles (base_footprint)          -> Collision Monitor
  2. desired velocity from the dodge schedule -> /cmd_vel_raw         -> Collision Monitor
  3. wait for /cmd_vel from Collision Monitor -> Watchdog (monitor answered + lidar fresh)
  4. velocity -> walk-these-ways policy -> joint targets -> MuJoCo (PD at 500 Hz)

The side to dodge is chosen at the trigger from the newest lidar frame (guard.choose_side).
Zero desired velocity never needs the monitor (it is always safe), so it bypasses it.

Runs under ROS 2 Humble in WSL with a private domain (ROS_LOCALHOST_ONLY=1, ROS_DOMAIN_ID=77)
so nothing reaches a real robot. Use test_avoid.py to run the scenario suite.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import go2_dodge_sim as g  # noqa: E402
import guard  # noqa: E402
from lidar_sim import SimLidar  # noqa: E402

FOOTPRINT = ((0.47, 0.30), (0.47, -0.30), (-0.47, -0.30), (-0.47, 0.30))   # Go2 incl. legs + 10 cm (test_avoid tuning)
NEED = 0.80            # lateral room required for the SAFE dodge (max travel 0.61 m + margin, dodge_travel.py)
CM_PARAMS = HERE / "collision_monitor.yaml"
MEASURE = 0.5          # clearance measurement range around obstacles [m]


@dataclass
class Obstacle:
    kind: str                       # "box" (size = half x, y, z) | "cyl" (size = radius, half height)
    pos: tuple                      # centre in world (robot starts at the origin facing +x)
    size: tuple
    appear: float | None = None     # pops into place this many seconds after the trigger


@dataclass
class Scenario:
    name: str
    obstacles: list = field(default_factory=list)
    phys: g.Physics = field(default_factory=g.Physics)
    lidar_off_at: float | None = None      # lidar stops publishing this long after the trigger
    monitor: str = "on"                    # "on" | "down" (no answers) | "bypass" (no monitor at all)
    lidar_watch: bool = True               # our watchdog's lidar-freshness check
    side_check: bool = True
    keep_off: bool = True                  # back away from obstacles closer than 12 cm after the dodge
    prefer: int = 1


@dataclass
class Outcome:
    name: str
    side: int = 0
    contact: bool = False
    contact_t: float | None = None
    min_clearance: float = float("inf")
    fell: bool = False
    stable_end: bool = False
    lateral: float = 0.0                   # signed displacement along the chosen side (or +y if none)
    t30: float | None = None
    limited_steps: int = 0                 # steps where the monitor reduced the velocity
    watchdog_zero_steps: int = 0
    keep_off_steps: int = 0
    room: tuple = (0.0, 0.0)
    seen: tuple = (False, False)
    fall_t: float | None = None
    notes: list = field(default_factory=list)
    trace: list = field(default_factory=list)      # (t_rel, wanted vy, commanded vy, actual vy, lateral)


def ros_env() -> dict:
    env = dict(os.environ)
    # CycloneDDS: FastDDS localhost-only discovery fails under WSL mirrored networking; the Go2 and
    # the Kria use CycloneDDS anyway. Private domain + localhost only: nothing reaches a real robot.
    env.update(ROS_DOMAIN_ID="77", ROS_LOCALHOST_ONLY="1", RMW_IMPLEMENTATION="rmw_cyclonedds_cpp",
               CYCLONEDDS_URI=f"file://{HERE / 'cyclonedds_wsl.xml'}")   # sub-MTU datagrams, see the file
    return env


class Bench:
    """ROS side: the Collision Monitor process plus our publishers/subscriber; one per test session."""

    def __init__(self, cm_params: Path = CM_PARAMS, log_path: Path | None = None):
        import rclpy
        from geometry_msgs.msg import Point32, PolygonStamped, Twist
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import PointCloud2

        os.environ.update(ros_env())
        self.log = open(log_path or HERE / "collision_monitor.log", "w")
        # the binary itself (not `ros2 run`, whose child can outlive it), after clearing stragglers:
        # two collision_monitor nodes on the test domain break the lifecycle activation
        binary = "/opt/ros/humble/lib/nav2_collision_monitor/collision_monitor"
        subprocess.run(["pkill", "-f", binary], check=False)
        time.sleep(0.3)
        self.cm = subprocess.Popen([binary, "--ros-args", "--params-file", str(cm_params)],
                                   stdout=self.log, stderr=subprocess.STDOUT, env=ros_env())
        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node("go2_avoid_bench")
        self.Twist, self.Clock = Twist, Clock
        self.pub_clock = self.node.create_publisher(Clock, "/clock", 10)
        self.pub_cloud = self.node.create_publisher(PointCloud2, "/avoid/cloud_obstacles", 10)
        self.pub_cmd = self.node.create_publisher(Twist, "/cmd_vel_raw", 10)
        self.pub_fp = self.node.create_publisher(PolygonStamped, "/avoid/footprint", 10)
        self.Point32, self.PolygonStamped = Point32, PolygonStamped
        self.last_out = None
        self.node.create_subscription(Twist, "/cmd_vel", self._on_out, 10)
        self.T = 1.0                       # ROS (sim) time; monotonic across scenarios
        self._activate()

    def _on_out(self, msg) -> None:
        self.last_out = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

    def clock(self, T: float) -> None:
        msg = self.Clock()
        msg.clock.sec, msg.clock.nanosec = int(T), int((T % 1) * 1e9)
        self.pub_clock.publish(msg)
        fp = self.PolygonStamped()          # the approach polygon only works with a published footprint
        fp.header.frame_id = "base_footprint"
        fp.header.stamp = msg.clock
        fp.polygon.points = [self.Point32(x=float(x), y=float(y), z=0.0) for x, y in FOOTPRINT]
        self.pub_fp.publish(fp)

    def _activate(self) -> None:
        from lifecycle_msgs.srv import ChangeState
        cli = self.node.create_client(ChangeState, "/collision_monitor/change_state")
        deadline = time.time() + 20
        while not cli.wait_for_service(timeout_sec=0.5):
            self.clock(self.T)
            if time.time() > deadline or self.cm.poll() is not None:
                raise RuntimeError("collision_monitor did not come up; see collision_monitor.log")
        for tid in (1, 3):                 # configure, activate
            req = ChangeState.Request()
            req.transition.id = tid
            fut = cli.call_async(req)
            while not fut.done():
                self.clock(self.T)
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if not fut.result().success:
                raise RuntimeError(f"lifecycle transition {tid} failed; see collision_monitor.log")

    def publish_cloud(self, stamp: float, pts: np.ndarray) -> None:
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header
        h = Header()
        h.frame_id = "base_footprint"
        h.stamp.sec, h.stamp.nanosec = int(stamp), int((stamp % 1) * 1e9)
        self.pub_cloud.publish(point_cloud2.create_cloud_xyz32(h, pts.astype(np.float32).tolist()))

    def ask(self, cmd: np.ndarray, timeout: float = 0.3) -> np.ndarray | None:
        """Send the desired velocity, return the monitor's answer (None if it does not answer)."""
        self.last_out = None
        msg = self.Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = map(float, cmd)
        self.pub_cmd.publish(msg)
        deadline = time.time() + timeout
        while self.last_out is None and time.time() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.002)
        return self.last_out

    def spin(self) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=0)

    def close(self) -> None:
        self.cm.terminate()
        try:
            self.cm.wait(5)
        except subprocess.TimeoutExpired:
            self.cm.kill()
        self.node.destroy_node()
        self.rclpy.shutdown()
        self.log.close()


def desired_vy(dodge: g.Dodge, t_rel: float) -> float:
    """SAFE dodge speed schedule (towards +side), seconds after the trigger."""
    if t_rel < 0:
        return 0.0
    if t_rel < dodge.dodge:
        return dodge.kick_vy if (dodge.kick_vy is not None and t_rel < dodge.kick_s) else dodge.cmd.vy
    if dodge.brake_vy is not None and t_rel < dodge.dodge + dodge.brake_s:
        return dodge.brake_vy
    return 0.0


def build(sc: Scenario):
    spec = g.flat_scene_spec()
    obs_geoms, mocap = [], {}
    for i, ob in enumerate(sc.obstacles):
        gtype = mujoco.mjtGeom.mjGEOM_BOX if ob.kind == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER
        size = list(ob.size) + [0] * (3 - len(ob.size))
        # margin = gap = MEASURE: MuJoCo reports every robot part within MEASURE of the obstacle as a
        # contact with its signed distance, but applies no force until real touch (dist <= 0).
        # (mj_geomDistance is unusable here: MuJoCo 3.3.7 returns 0.0 for box-box pairs 0.6 m apart.)
        kw = dict(name=f"obs{i}", type=gtype, size=size, margin=MEASURE, gap=MEASURE)
        if ob.appear is None:
            spec.worldbody.add_geom(pos=list(ob.pos), rgba=(0.8, 0.3, 0.2, 1), **kw)
        else:
            b = spec.worldbody.add_body(name=f"pop{i}", mocap=True, pos=[40.0 + 3 * i, 40.0, ob.pos[2]])
            b.add_geom(rgba=(0.9, 0.6, 0.1, 1), **kw)
    m = spec.compile()
    for i, ob in enumerate(sc.obstacles):
        obs_geoms.append(m.geom(f"obs{i}").id)
        if ob.appear is not None:
            mocap[i] = m.body_mocapid[m.body(f"pop{i}").id]
    return m, obs_geoms, mocap


def run(sc: Scenario, bench: Bench | None, policy: g.WTWPolicy, dodge: g.Dodge) -> Outcome:
    out = Outcome(sc.name)
    m, obs_geoms, mocap = build(sc)
    m.opt.timestep = sc.phys.timestep
    d = mujoco.MjData(m)
    r = g.Go2Robot(m, d, sc.phys)
    mujoco.mj_resetData(m, d)
    r.reset()
    mujoco.mj_forward(m, d)
    robot_bodies = {b for b in range(m.nbody) if b == r.trunk or g._below(m, b, r.trunk)}
    lidar = SimLidar(m, d, r.trunk, seed=sc.phys.seed + 1)
    to_fp, hist = guard.LidarToFootprint(), guard.StateHistory()
    wd = guard.Watchdog()
    if not sc.lidar_watch:
        wd.max_cloud_age = 1e9
    policy.reset()
    latest = (np.zeros((0, 3)), np.zeros((0, 3)))
    recent = guard.RecentFrames(3)
    T0 = bench.T if bench else 0.0
    t0 = dodge.t0 + sc.phys.pre_offset
    t1 = t0 + dodge.dodge + dodge.brake
    total = t1 + dodge.stand
    substeps = round(g.POLICY_DT / m.opt.timestep)
    stand_from = None
    start_xy = None
    pace = None
    want = cmd = np.zeros(3)
    t = 0.0
    while t < total - 1e-9:
        T = T0 + t
        if bench:
            bench.clock(T)
        # ---- decide the command -----------------------------------------------------------------
        if t < dodge.settle:
            q_t, kp, kd = g.DEFAULT_Q, dodge.stand_kp, dodge.stand_kd
        elif t < t1:
            if t >= t0 and start_xy is None:                       # trigger
                start_xy = r.pos[:2].copy()
                left = guard.side_room(*recent.merged(), +1, NEED)
                right = guard.side_room(*recent.merged(), -1, NEED)
                out.room, out.seen = (left[0], right[0]), (left[1], right[1])
                out.side = guard.choose_side(left, right, NEED, sc.prefer) if sc.side_check else sc.prefer
                if out.side == 0:
                    out.notes.append("no side with room: no dodge")
            want = np.array([0.0, out.side * desired_vy(dodge, t - t0), 0.0]) if t >= t0 else np.zeros(3)
            keepoff = False
            if sc.keep_off and t >= t0 and not want.any():          # trotting in place after the dodge
                vy_off = guard.keep_off(latest[0], out.side)
                if vy_off:
                    want, keepoff = np.array([0.0, vy_off, 0.0]), True
                    out.keep_off_steps += 1
            if not want.any():
                cmd = np.zeros(3)
            elif keepoff:          # around the monitor (it blocks all motion once inside its footprint)
                cmd = want if wd.command_fresh_lidar(T) else np.zeros(3)
            elif sc.monitor == "bypass":
                cmd = want
            else:
                ans = bench.ask(want) if (bench and sc.monitor == "on") else None
                if ans is not None:
                    wd.monitor(T, ans)
                cmd = wd.command(T)
                if not cmd.any():
                    out.watchdog_zero_steps += 1      # zero from the watchdog or from the monitor itself
                    if wd.reason != "ok" and wd.reason.split(" (")[0] not in [n.split(" (")[0] for n in out.notes]:
                        out.notes.append(wd.reason)
                elif np.linalg.norm(cmd) < np.linalg.norm(want) - 1e-3:
                    out.limited_steps += 1
            c = replace(dodge.cmd, vx=float(cmd[0]), vy=float(cmd[1]), yaw=float(cmd[2]))
            q_t, kp, kd = policy.targets(*r.observe(), c), g.KP, g.KD
        else:
            if stand_from is None:
                stand_from = r.q
            a = min(1.0, (t - t1) / dodge.stand_ramp)
            q_t, kp, kd = (1 - a) * stand_from + a * g.DEFAULT_Q, dodge.stand_kp, dodge.stand_kd
        # ---- events ---------------------------------------------------------------------------------
        for i, ob in enumerate(sc.obstacles):
            if ob.appear is not None and t >= t0 + ob.appear:
                d.mocap_pos[mocap[i]] = ob.pos
        if sc.lidar_off_at is not None and t >= t0 + sc.lidar_off_at:
            lidar.enabled = False
        # ---- physics + lidar ------------------------------------------------------------------------
        for _ in range(substeps):
            r.apply_pd(q_t, kp, kd)
            mujoco.mj_step(m, d)
            t += m.opt.timestep
            hist.add(t, d.qpos[r.bq + 3:r.bq + 7], r.q)
            for stamp, pts in lidar.update(t):
                st = hist.at(stamp)
                if st is None:
                    continue
                quat, q = st
                p_fp = to_fp(pts, quat, q)
                latest = guard.self_and_floor(p_fp, q, quat)
                recent.add(*latest)
                wd.cloud(T0 + stamp)
                if bench:
                    bench.publish_cloud(T0 + stamp, latest[0])
            for k in range(d.ncon):          # obstacle contacts carry the signed distance (see build)
                c = d.contact[k]
                c1, c2 = c.geom1, c.geom2
                if (c1 in obs_geoms and m.geom_bodyid[c2] in robot_bodies) or \
                        (c2 in obs_geoms and m.geom_bodyid[c1] in robot_bodies):
                    out.min_clearance = min(out.min_clearance, float(c.dist))
                    if c.dist <= 0.0 and not out.contact:
                        out.contact, out.contact_t = True, t - t0
        # ---- measurements (per control step) ---------------------------------------------------------
        rpy = r.rpy()
        if r.pos[2] < 0.15 or abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8 or r.trunk_on_floor():
            if not out.fell:
                out.fall_t = t - t0
            out.fell = True
        if t0 - 0.1 <= t <= t1 + 0.5:
            side = out.side or 1
            out.trace.append((round(t - t0, 3), round(float(want[1]) if t < t1 and t >= t0 else 0.0, 3),
                              round(float(cmd[1]) if t < t1 and t >= t0 else 0.0, 3),
                              round(float(side * r.vel[1]), 3), round(out.lateral, 3)))
        if start_xy is not None:
            lat = (out.side or 1) * (r.pos[1] - start_xy[1])
            out.lateral = lat
            if out.t30 is None and lat >= 0.30:
                out.t30 = t - t0
        if bench:
            bench.spin()
            # Real-time pacing from 0.5 s before the trigger to the end of braking: the monitor
            # subscribes best-effort, and faster-than-real-time bursts make DDS drop most clouds
            # (measured: 15 of 118 delivered). The real lidar arrives at a steady 11 Hz.
            if t0 - 0.5 <= t <= t1:
                if pace is None:
                    pace = (time.perf_counter(), t)
                while time.perf_counter() < pace[0] + (t - pace[1]):
                    bench.rclpy.spin_once(bench.node, timeout_sec=0.001)
    rpy = r.rpy()
    out.stable_end = (not out.fell) and int(r.foot_contacts().sum()) == 4 and \
        abs(math.degrees(rpy[0])) < 5 and abs(math.degrees(rpy[1])) < 5
    if bench:
        bench.T = T0 + t + 1.0
    return out
