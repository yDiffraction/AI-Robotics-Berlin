#!/usr/bin/env python3
"""Time the two-stage reflex in the Gazebo demo (run while `./sim.sh demo2` is up).

Listens to /reflex/event (stage 1 = armed on a wind-up, stage 2 = dash or abort) and
to the robot's ground-truth position, and reports per trial how long the robot needed
from each stage until it had moved.

All times are simulation time (Gazebo may run slower than real time; the factor is
printed). Distances are along the heading the robot had at stage 1, forward positive.

  ./sim.sh measure --trials 12
  ./sim.sh measure --delays 0.05,0.1,0.3,0.6 --trials 5    # with demo2 keyboard:=true

With --delays the tool triggers the reflex itself: arm, dash after the given delay, wait
until the robot is home again; every delay `--trials` times.
"""

import argparse
import bisect
import json
import math
import statistics
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Empty, String


MOVING = 0.02  # m from the start that count as "the robot moves"
HOME = 0.03    # m from the start that count as "back home"
FALLEN = 0.8   # rad of body tilt that count as "fell over"


class Recorder(Node):
    def __init__(self, odom_topic):
        super().__init__("measure_reflex")
        self.wall, self.sim, self.xy, self.yaw, self.tilt = [], [], [], [], []
        self.events = []
        self.create_subscription(Odometry, odom_topic, self.on_odom, 1000)
        self.create_subscription(String, "/reflex/event", self.on_event, 100)
        self.create_subscription(String, "/throw_scene/event", self.on_event, 100)

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.wall.append(time.time_ns() * 1e-9)
        self.sim.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.xy.append((p.x, p.y))
        self.yaw.append(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        self.tilt.append(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))  # body z axis against up

    def on_event(self, msg):
        self.events.append(json.loads(msg.data))

    def sim_time(self, wall_ns):
        """Simulation time at a wall-clock time, interpolated between odometry samples."""
        t = wall_ns * 1e-9
        i = min(max(bisect.bisect_left(self.wall, t), 1), len(self.wall) - 1)
        w0, w1, s0, s1 = self.wall[i - 1], self.wall[i], self.sim[i - 1], self.sim[i]
        return s0 + (s1 - s0) * (t - w0) / (w1 - w0) if w1 > w0 else s0


def first_crossing(times, values, start, level, rising):
    """Time after `start` at which the values first pass `level` (upwards if `rising`), linearly interpolated."""
    for i in range(bisect.bisect_left(times, start), len(times)):
        if (values[i] >= level) == rising:
            if i == 0 or values[i] == values[i - 1]:
                return times[i]
            share = (level - values[i - 1]) / (values[i] - values[i - 1])
            return max(start, times[i - 1] + min(max(share, 0.0), 1.0) * (times[i] - times[i - 1]))
    return None


def analyse(rec, trial):
    """Timing of one trial: a list of its /reflex/event messages, in order."""
    by_name = {}
    for e in trial:
        by_name.setdefault(e["event"], e)
    t_windup = rec.sim_time(by_name["armed"]["wall_time_ns"])
    decision = by_name.get("dash") or by_name.get("return")
    if decision is None or "idle" not in by_name:
        return None
    t_decision = rec.sim_time(decision["wall_time_ns"])
    t_end = rec.sim_time(by_name["idle"]["wall_time_ns"])

    i0 = bisect.bisect_left(rec.sim, t_windup)
    (x0, y0), yaw = rec.xy[i0], rec.yaw[i0]
    lo, hi = i0, bisect.bisect_right(rec.sim, t_end)
    t = rec.sim[lo:hi]
    d = [(x - x0) * math.cos(yaw) + (y - y0) * math.sin(yaw) for x, y in rec.xy[lo:hi]]
    at = lambda when: d[min(bisect.bisect_left(t, when), len(d) - 1)]  # noqa: E731
    d_decision = at(t_decision)

    out = {"trial": by_name["armed"]["trial"], "kind": "throw" if "dash" in by_name else "abort",
           "windup_to_decision": t_decision - t_windup, "lean_in": d_decision,
           "speed_at_decision": (d_decision - at(t_decision - 0.1)) / 0.1}
    moving = first_crossing(t, [abs(v) for v in d], t_windup, MOVING, True)
    out["windup_to_moving"] = moving - t_windup if moving else None
    if out["kind"] == "throw":
        # reverse: back behind the point where the decision came
        for name, back in (("reverse", 0.005), ("back_10cm", 0.1), ("back_25cm", 0.25), ("back_50cm", 0.5)):
            hit = first_crossing(t, d, t_decision, d_decision - back, False)
            out[f"decision_to_{name}"] = hit - t_decision if hit else None
        out["windup_to_back_50cm"] = (out["decision_to_back_50cm"] + out["windup_to_decision"]
                                      if out["decision_to_back_50cm"] else None)
        out["furthest_back"] = min(d) - d_decision
    else:
        home = first_crossing(t, [abs(v) for v in d], t_decision, HOME, False)
        out["decision_to_home"] = home - t_decision if home is not None else None
    out["end_offset"] = d[-1]
    out["max_tilt"] = max(rec.tilt[lo:hi])
    return out


def spin_until(rec, done, timeout):
    """Spin until done() is true; False if that takes longer than `timeout` seconds."""
    end = time.monotonic() + timeout
    while rclpy.ok() and not done():
        if time.monotonic() > end:
            return False
        rclpy.spin_once(rec, timeout_sec=0.002)
    return True


def drive(rec, delays, timeout):
    """Trigger the reflex ourselves: arm, dash after each delay, wait until the robot is home. Returns trial -> delay."""
    arm, dash = rec.create_publisher(Empty, "/reflex/arm", 10), rec.create_publisher(Empty, "/reflex/dash", 10)
    end = time.monotonic() + timeout
    if not spin_until(rec, lambda: arm.get_subscription_count() and dash.get_subscription_count() and len(rec.sim) > 1, 30):
        return {}
    commanded = {}
    for delay in delays:
        seen = len(rec.events)
        start = time.monotonic()
        arm.publish(Empty())
        spin_until(rec, lambda: time.monotonic() - start >= delay, delay + 1)
        dash.publish(Empty())
        new = lambda name: [e for e in rec.events[seen:] if e["event"] == name and "trial" in e]  # noqa: E731
        if not spin_until(rec, lambda: new("idle"), 20) or not new("armed"):
            print(f"no finished trial after arm + dash ({delay * 1e3:.0f} ms) - is demo2 keyboard:=true running?")
            break
        commanded[new("armed")[0]["trial"]] = delay
        pause = time.monotonic() + 1.0
        spin_until(rec, lambda: time.monotonic() > pause, 2)
        if rec.tilt[-1] > FALLEN:
            print(f"the robot fell over in trial {len(commanded)} (dash {delay * 1e3:.0f} ms after arm), stopping")
            break
        if time.monotonic() > end:
            break
    return commanded


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=10, help="finished trials to wait for (with --delays: per delay)")
    p.add_argument("--delays", help="trigger the reflex from here: comma-separated times between arm and dash in s")
    p.add_argument("--timeout", type=float, default=300.0, help="give up after this many seconds")
    p.add_argument("--odom", default="/odom/ground_truth_fast", help="position topic (nav_msgs/Odometry)")
    p.add_argument("--json", action="store_true", help="also print the raw numbers as one JSON line")
    args = p.parse_args()

    rclpy.init()
    rec = Recorder(args.odom)
    commanded = {}
    if args.delays:
        commanded = drive(rec, [float(d) for d in args.delays.split(",")] * args.trials, args.timeout)
    else:
        finished = lambda: sum(e["event"] == "idle" and "trial" in e for e in rec.events)  # noqa: E731
        spin_until(rec, lambda: finished() >= args.trials + 1, args.timeout)
    if len(rec.sim) < 2:
        raise SystemExit(f"no position on {args.odom} - is ./sim.sh demo2 running?")

    trials = {}
    for e in rec.events:
        if "trial" in e:
            trials.setdefault(e["trial"], []).append(e)
    # the first trial may have started before this tool did
    results = [r for n, ev in sorted(trials.items()) if ev[0]["event"] == "armed" and (r := analyse(rec, ev))]
    scene = [e for e in rec.events if e["event"] == "judged"]

    rate = (rec.sim[-1] - rec.sim[0]) / (rec.wall[-1] - rec.wall[0])
    odom_hz = (len(rec.sim) - 1) / (rec.sim[-1] - rec.sim[0])
    print(f"real-time factor {rate:.2f}, position at {odom_hz:.0f} Hz (sim time), {len(results)} trials")

    def ms(v):
        return "    -" if v is None else f"{v * 1e3:5.0f}"

    print("\nthrows   [ms]       arm->     arm->  lean-in  speed at  decision-> decision-> decision-> decision->     arm->  furthest   max tilt")
    print("trial            moving    decision   [cm]    decision    reverse    -10 cm     -25 cm     -50 cm     -50 cm    back [cm]    [deg]")
    throws = [r for r in results if r["kind"] == "throw"]
    for r in throws:
        print(f"{r['trial']:5d}          {ms(r['windup_to_moving'])}     {ms(r['windup_to_decision'])}    {r['lean_in'] * 100:5.1f}"
              f"    {r['speed_at_decision']:+.2f} m/s    {ms(r['decision_to_reverse'])}      {ms(r['decision_to_back_10cm'])}"
              f"      {ms(r['decision_to_back_25cm'])}      {ms(r['decision_to_back_50cm'])}     {ms(r['windup_to_back_50cm'])}"
              f"     {r['furthest_back'] * 100:6.1f}     {math.degrees(r['max_tilt']):5.1f}{'  FELL' if r['max_tilt'] > FALLEN else ''}")

    def median(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return statistics.median(values) if values else None

    if throws:
        print(f"median         {ms(median(throws, 'windup_to_moving'))}     {ms(median(throws, 'windup_to_decision'))}"
              f"    {median(throws, 'lean_in') * 100:5.1f}    {median(throws, 'speed_at_decision'):+.2f} m/s"
              f"    {ms(median(throws, 'decision_to_reverse'))}      {ms(median(throws, 'decision_to_back_10cm'))}"
              f"      {ms(median(throws, 'decision_to_back_25cm'))}      {ms(median(throws, 'decision_to_back_50cm'))}"
              f"     {ms(median(throws, 'windup_to_back_50cm'))}     {median(throws, 'furthest_back') * 100:6.1f}"
              f"     {math.degrees(median(throws, 'max_tilt')):5.1f}")
    if commanded:
        print("\nmedian per delay [ms]       lean-in  speed at  decision-> decision-> decision-> decision->     arm->   max tilt")
        print("arm->dash       trials      [cm]    decision    reverse    -10 cm     -25 cm     -50 cm     -50 cm    [deg]   fell")
        for delay in sorted(set(commanded.values())):
            rows = [r for r in throws if commanded.get(r["trial"]) == delay]
            if rows:
                print(f"{delay * 1e3:6.0f}          {len(rows):4d}       {median(rows, 'lean_in') * 100:5.1f}"
                      f"    {median(rows, 'speed_at_decision'):+.2f} m/s    {ms(median(rows, 'decision_to_reverse'))}"
                      f"      {ms(median(rows, 'decision_to_back_10cm'))}      {ms(median(rows, 'decision_to_back_25cm'))}"
                      f"      {ms(median(rows, 'decision_to_back_50cm'))}     {ms(median(rows, 'windup_to_back_50cm'))}"
                      f"    {math.degrees(max(r['max_tilt'] for r in rows)):5.1f}   {sum(r['max_tilt'] > FALLEN for r in rows):3d}")

    aborts = [r for r in results if r["kind"] == "abort"]
    print("\naborts   [ms]       arm->     arm->  lean-in  decision->  end offset")
    print("trial            moving    decision   [cm]      home        [cm]")
    for r in aborts:
        print(f"{r['trial']:5d}          {ms(r['windup_to_moving'])}     {ms(r['windup_to_decision'])}    {r['lean_in'] * 100:5.1f}"
              f"     {ms(r['decision_to_home'])}      {r['end_offset'] * 100:+6.1f}")
    if aborts:
        print(f"median         {ms(median(aborts, 'windup_to_moving'))}     {ms(median(aborts, 'windup_to_decision'))}"
              f"    {median(aborts, 'lean_in') * 100:5.1f}     {ms(median(aborts, 'decision_to_home'))}"
              f"      {median(aborts, 'end_offset') * 100:+6.1f}")

    if scene:
        dodged = sum("DODGED" in e["result"] for e in scene)
        print(f"\nvirtual thrower: {dodged} of {len(scene)} balls dodged")
    if args.json:
        print(json.dumps({"real_time_factor": rate, "trials": results, "scene": scene}))
    rec.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
