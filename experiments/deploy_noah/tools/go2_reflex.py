"""Two-stage dodge on the real Go2, driven from the laptop over the LAN cable (no ROS, no Docker).

Same moves as sim/ros_ws/.../two_stage_reflex.py, triggered from the keyboard:

  a      arm     the robot leans in a little and stays armed for a moment
  d      dash    while armed: the robot backs off, waits and walks back to the start
  s      disarm  while armed: the robot walks back to the start
  space  STOP    stop at once (StopMove), back to idle where it stands
  q      quit    (also Ctrl+C), sends StopMove before leaving

Talks straight to the Go2 sport service: Move (1008) / StopMove (1003) on rt/api/sport/request,
DDS domain 0, over the given network interface. Distances come from the robot's own odometry
(rt/sportmodestate); the robot does not move before that state arrives.

Laptop setup (the Go2 network is 192.168.123.0/24, Go2 = .161, Kria = .99):
  nmcli con add type ethernet ifname enx9cbf0d013571 con-name go2 ipv4.method manual \
      ipv4.addresses 192.168.123.51/24 ipv6.method disabled
  nmcli con up go2 && ping -c 3 192.168.123.161

Run (conda env "durance", see sim/mujoco/README.md):
  python tools/go2_reflex.py --iface enx9cbf0d013571            # real robot
  python tools/go2_reflex.py --dry-run                          # print the requests, no DDS
  python tools/go2_reflex.py --domain 1 --iface lo              # against a sim on domain 1

No lidar check here: keep the space in front of and behind the robot free.
"""

import argparse
import json
import math
import os
import select
import subprocess
import sys
import termios
import time
import tty

MOVE, STOP_MOVE = 1008, 1003
IDLE, ARMED, DASH, HOLD, RETURN = "idle", "armed", "dash", "hold", "return"
TICK = 0.02
RESEND = 0.1  # the Go2 drops a velocity that is not refreshed
STATE_TIMEOUT = 0.5  # stop when the robot's state has not come for this long
TEXT = {"armed": "ARMED - d dash, s disarm", "dash": "DASH", "hold": "dodged, waiting",
        "return": "walking back to the start", "idle": "at the start - a to arm"}


class Link:
    """Sport requests out, robot state and responses in."""

    def __init__(self, domain, iface, state_topic):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_api.msg.dds_ import (Request_, RequestHeader_, RequestIdentity_,
                                                             RequestLease_, RequestPolicy_, Response_)
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        self.types = Request_, RequestHeader_, RequestIdentity_, RequestLease_, RequestPolicy_
        ChannelFactoryInitialize(domain, iface)
        self.pub = ChannelPublisher("rt/api/sport/request", Request_)
        self.pub.Init()
        self.pose, self.pose_time, self.responses = None, 0.0, []
        self.state_sub = ChannelSubscriber(state_topic, SportModeState_)
        self.state_sub.Init(self.on_state, 1)
        self.resp_sub = ChannelSubscriber("rt/api/sport/response", Response_)
        self.resp_sub.Init(lambda m: self.responses.append((m.header.identity.api_id, m.header.status.code)), 10)

    def on_state(self, msg):
        self.pose = (msg.position[0], msg.position[1], msg.imu_state.rpy[2])
        self.pose_time = time.monotonic()

    def send(self, api_id, parameter, no_reply):
        # built by hand: unitree_sdk2py's default Request_ helper raises a TypeError
        Request_, RequestHeader_, RequestIdentity_, RequestLease_, RequestPolicy_ = self.types
        identity = RequestIdentity_(time.monotonic_ns(), api_id)
        header = RequestHeader_(identity, RequestLease_(0), RequestPolicy_(0, no_reply))
        self.pub.Write(Request_(header, parameter, []))


class DryLink:
    """Prints the requests and fakes the odometry from the commanded speed."""

    def __init__(self):
        self.pose, self.pose_time, self.responses = (0.0, 0.0, 0.0), time.monotonic(), []
        self.speed, self.last = 0.0, time.monotonic()

    def send(self, api_id, parameter, no_reply):
        self.speed = json.loads(parameter)["x"] if api_id == MOVE else 0.0
        if api_id != MOVE or not hasattr(self, "printed") or self.printed != parameter:
            print(f"    -> api_id {api_id} {parameter}")
        self.printed = parameter

    def advance(self):
        now = time.monotonic()
        self.pose = (self.pose[0] + self.speed * (now - self.last), 0.0, 0.0)
        self.pose_time, self.last = now, now


class Reflex:
    def __init__(self, link, args, log):
        self.link, self.a, self.log = link, args, log
        self.state, self.since = IDLE, time.monotonic()
        self.origin, self.target, self.speed, self.sent, self.trial = None, 0.0, 0.0, 0.0, 0

    # --- helpers -------------------------------------------------------------
    def drive(self, speed):
        """Set the forward speed; 0 stops."""
        self.speed, self.sent = speed, time.monotonic()
        if speed:
            self.link.send(MOVE, json.dumps({"x": speed, "y": 0.0, "z": 0.0}), True)
        else:
            self.link.send(STOP_MOVE, "{}", False)

    def distance(self):
        """Metres moved forward since the robot was armed, along the heading it had then."""
        if self.link.pose is None or self.origin is None:
            return 0.0
        x0, y0, yaw = self.origin
        x, y, _ = self.link.pose
        return (x - x0) * math.cos(yaw) + (y - y0) * math.sin(yaw)

    def event(self, name, **data):
        data.update(event=name, trial=self.trial, wall_time_ns=time.time_ns(), distance=round(self.distance(), 3))
        if self.log:
            self.log.write(json.dumps(data) + "\n")
            self.log.flush()
        if name == "ignored":
            print(f"  {data['trigger']} ignored: state '{data['state']}'")
        else:
            reason = f" ({data['reason']})" if "reason" in data else ""
            print(f"  {TEXT.get(name, name)}{reason}   [{data['distance'] * 100:+.0f} cm]")

    def enter(self, state, **data):
        self.state, self.since = state, time.monotonic()
        self.event(state, **data)

    def fresh(self):
        return time.monotonic() - self.link.pose_time < STATE_TIMEOUT

    # --- inputs --------------------------------------------------------------
    def arm(self):
        if self.state != IDLE:
            return self.event("ignored", trigger="arm", state=self.state)  # not back at the start yet
        if not self.fresh():
            return print("  arm ignored: no robot state (rt/sportmodestate) - cable, IP, domain?")
        self.trial += 1
        self.origin, self.target = self.link.pose, self.a.approach_dist
        self.enter(ARMED)
        if self.a.approach_speed > 0:
            self.drive(self.a.approach_speed)

    def dash(self):
        if self.state != ARMED:
            return self.event("ignored", trigger="dash", state=self.state)
        self.target = self.distance() - self.a.retreat_dist
        self.enter(DASH)
        self.drive(-self.a.retreat_speed)

    def disarm(self):
        if self.state == ARMED:
            self.go_home("disarm")

    def stop(self, reason="stop key"):
        self.drive(0.0)
        if self.state != IDLE:
            self.enter(IDLE, reason=reason)

    # --- behaviour -----------------------------------------------------------
    def go_home(self, reason):
        self.target = 0.0
        self.enter(RETURN, reason=reason)
        d = self.distance()
        self.drive(math.copysign(self.a.return_speed, -d) if abs(d) > self.a.tolerance else 0.0)

    def tick(self):
        now = time.monotonic()
        d, elapsed = self.distance(), now - self.since
        if self.speed and not self.fresh():
            return self.stop("robot state lost")

        if self.state == ARMED:
            if self.speed and d >= self.target:
                self.drive(0.0)                      # leaned in far enough, wait for the decision
            if elapsed > self.a.arm_time:
                self.go_home("timeout")
        elif self.state == DASH:
            if d <= self.target or elapsed > 2 * self.a.retreat_dist / self.a.retreat_speed + 1.0:
                self.drive(0.0)
                self.enter(HOLD)
        elif self.state == HOLD:
            if elapsed > self.a.hold_time:
                self.go_home("after dash")
        elif self.state == RETURN:
            done = d <= self.target if self.speed < 0 else d >= self.target
            limit = 2 * (self.a.retreat_dist + self.a.approach_dist) / self.a.return_speed + 1.0
            if not self.speed or done or elapsed > limit:
                self.drive(0.0)
                self.enter(IDLE)
        if self.speed and now - self.sent > RESEND:
            self.drive(self.speed)


def check_iface(iface):
    """Warn when the interface has no address on the Go2 network."""
    if iface == "lo":
        return
    out = subprocess.run(["ip", "-4", "-br", "addr", "show", "dev", iface], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"no interface {iface!r} (ip -br addr shows the names)")
    if "192.168.123." not in out.stdout:
        print(f"WARNING: {iface} has no 192.168.123.x address: {out.stdout.strip() or 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iface", default="enx9cbf0d013571", help="network interface to the Go2")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain (real Go2: 0, MuJoCo sim: 1)")
    ap.add_argument("--state-topic", default="rt/sportmodestate")
    ap.add_argument("--dry-run", action="store_true", help="print the requests, no DDS")
    ap.add_argument("--arm-time", type=float, default=1.5, help="s the robot stays armed")
    ap.add_argument("--approach-speed", type=float, default=0.2, help="m/s forward when armed, 0: stay put")
    ap.add_argument("--approach-dist", type=float, default=0.10, help="m forward at most when armed")
    ap.add_argument("--retreat-speed", type=float, default=0.6, help="m/s backwards on the dash")
    ap.add_argument("--retreat-dist", type=float, default=0.4, help="m backwards on the dash")
    ap.add_argument("--return-speed", type=float, default=0.25, help="m/s back to the start")
    ap.add_argument("--hold-time", type=float, default=1.0, help="s to wait after the dash")
    ap.add_argument("--tolerance", type=float, default=0.03, help="m, closer than this counts as at the start")
    ap.add_argument("--log", help="append the events as JSON lines to this file")
    args = ap.parse_args()
    if not 0 < args.retreat_speed <= 1.5 or not 0 <= args.approach_speed <= 0.5 or not 0 < args.return_speed <= 0.5:
        raise SystemExit("speeds out of range: retreat 0..1.5, approach 0..0.5, return 0..0.5 m/s")

    if args.dry_run:
        link = DryLink()
    else:
        check_iface(args.iface)
        link = Link(args.domain, args.iface, args.state_topic)
    reflex = Reflex(link, args, open(args.log, "a") if args.log else None)

    print(f"domain {args.domain} on {args.iface}" if not args.dry_run else "dry run, nothing is sent")
    print(f"approach {args.approach_speed} m/s up to {args.approach_dist} m, retreat {args.retreat_speed} m/s "
          f"for {args.retreat_dist} m, back at {args.return_speed} m/s")
    if not args.dry_run:
        deadline = time.monotonic() + 3.0
        while not reflex.fresh() and time.monotonic() < deadline:
            time.sleep(0.05)
        print("robot state OK" if reflex.fresh() else
              f"no {args.state_topic} yet - check cable, IP and that the Go2 is on (arming waits for it)")
    print(__doc__.split("\n\n")[2])

    tty_in = sys.stdin.isatty()
    settings = termios.tcgetattr(sys.stdin) if tty_in else None
    keys = {"a": reflex.arm, "d": reflex.dash, "s": reflex.disarm, " ": reflex.stop}
    seen, next_tick = 0, time.monotonic()
    try:
        if tty_in:
            tty.setcbreak(sys.stdin.fileno())
        while True:
            if args.dry_run:
                link.advance()
            while select.select([sys.stdin], [], [], 0)[0]:
                key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()
                if key in ("q", ""):  # "" = end of piped input
                    raise KeyboardInterrupt
                if key in keys:
                    keys[key]()
            reflex.tick()
            for api_id, code in link.responses[seen:]:
                if code:
                    print(f"  robot answered api_id {api_id} with error code {code}")
            seen = len(link.responses)
            next_tick += TICK
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            link.send(STOP_MOVE, "{}", False)
            time.sleep(0.05)
        if tty_in:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print("stopped")


if __name__ == "__main__":
    main()
