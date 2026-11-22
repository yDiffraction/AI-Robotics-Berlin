#!/usr/bin/env python3
"""Dodge only: the Go2 jumps out of the way when a trigger arrives, then walks slowly back to where it was.

Triggers:
  X button of the Go2 remote   (/wirelesscontroller, unitree_go/msg/WirelessController, keys bit 10; on press)
  /reflex/dash                 ros2 topic pub --once /reflex/dash std_msgs/msg/Empty "{}"   (also dodge_keys.py)
  /dodge/stop                  ros2 topic pub --once /dodge/stop std_msgs/msg/Empty "{}"    stop at once (StopMove)

Dodge: sport API Move (api_id 1008) at `speed` m/s (default 1.1) in `direction` (default back) for `duration` s,
the velocity re-sent every 0.1 s (the Go2 drops a velocity that is not refreshed), StopMove (1003), wait `hold` s,
then walk back at `return_speed` (default 0.25 m/s) until the robot is at its start position again. The way back
uses the robot's odometry (`odom`: sportmodestate = /sportmodestate, odometry = nav_msgs/Odometry on `odom_topic`,
e.g. in the Gazebo sim); without fresh odometry it walks back `return_scale` (0.8) of the commanded distance by
time, so it never overshoots towards the thrower. Messages go to /api/sport/request (unitree_api/msg/Request),
like the starter's action_bridge.

Safety: speeds and durations are clamped; a trigger is ignored while a dodge runs and for `refractory` s after it;
every phase has a time limit; /dodge/stop and Ctrl+C send StopMove (3 times on exit); `dry_run:=true` only logs.
There is no obstacle check: keep the space behind the robot free.

Run (ROS 2 Humble + the go2-kria overlay with unitree_api and unitree_go, Go2 on DDS domain 0):
  python3 go2_dodge.py --ros-args -p dry_run:=true       # log only, nothing sent
  python3 go2_dodge.py                                   # X on the remote: back 1.1 m/s for 0.5 s, then slowly back
  python3 go2_dodge.py --ros-args -p speed:=0.8 -p return_speed:=0.3
  python3 go2_dodge.py --ros-args -p odom:=odometry -p odom_topic:=/odom/ground_truth_fast   # Gazebo sim
Status: /dodge/event (std_msgs/String, JSON): dash, hold, return, done (with the measured distances), stop, ignored.
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Empty, String
from unitree_api.msg import Request

MOVE, STOP_MOVE = 1008, 1003
DIRS = {"back": (-1.0, 0.0), "forward": (1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
# bits of WirelessController.keys (unitree_sdk2: R1 L1 start select R2 L2 F1 F2 A B X Y up right down left)
BUTTONS = {"r1": 0, "l1": 1, "start": 2, "select": 3, "r2": 4, "l2": 5, "f1": 6, "f2": 7,
           "a": 8, "b": 9, "x": 10, "y": 11, "up": 12, "right": 13, "down": 14, "left": 15}
IDLE, OUT, HOLD, RETURN = "idle", "dash", "hold", "return"
ODOM_TIMEOUT = 0.5


def clamp(v, lo, hi):
    return min(max(float(v), lo), hi)


class Go2Dodge(Node):
    def __init__(self):
        super().__init__("go2_dodge")
        p = self.declare_parameter
        self.direction = p("direction", "back").value
        self.speed = clamp(p("speed", 1.1).value, 0.1, 1.5)               # m/s away from the start
        self.duration = clamp(p("duration", 0.5).value, 0.1, 2.0)         # s at that speed
        self.come_back = p("come_back", True).value
        self.hold = clamp(p("hold", 1.0).value, 0.0, 5.0)                 # s standing before walking back
        self.return_speed = clamp(p("return_speed", 0.25).value, 0.1, 0.5)
        self.return_scale = clamp(p("return_scale", 0.8).value, 0.0, 1.0)  # share walked back without odometry
        self.tolerance = clamp(p("tolerance", 0.05).value, 0.01, 0.3)     # m, closer counts as back at the start
        self.refractory = clamp(p("refractory", 1.0).value, 0.0, 10.0)
        self.dry_run = p("dry_run", False).value
        self.button = p("button", "x").value.lower()                      # remote button, "none" = off
        odom = p("odom", "sportmodestate").value                          # sportmodestate | odometry | none
        odom_topic = p("odom_topic", "").value
        controller_topic = p("controller_topic", "/wirelesscontroller").value
        if self.direction not in DIRS:
            raise SystemExit("direction must be one of %s" % ", ".join(DIRS))
        if self.button != "none" and self.button not in BUTTONS:
            raise SystemExit("button must be none or one of %s" % ", ".join(BUTTONS))

        self.pub = self.create_publisher(Request, "/api/sport/request", 10)
        self.event_pub = self.create_publisher(String, "/dodge/event", 10)
        for topic in ("/reflex/dash", "/dodge/dash"):
            self.create_subscription(Empty, topic, lambda _m, t=topic: self.on_dash(t), 10)
        for topic in ("/dodge/stop", "/reflex/stop"):
            self.create_subscription(Empty, topic, lambda _m: self.stop("stop topic"), 10)
        if self.button != "none":
            from unitree_go.msg import WirelessController
            self.create_subscription(WirelessController, controller_topic, self.on_remote, qos_profile_sensor_data)
        self.pose, self.pose_time = None, 0.0                             # (x, y, yaw) from the odometry
        if odom == "sportmodestate":
            from unitree_go.msg import SportModeState
            self.create_subscription(SportModeState, odom_topic or "/sportmodestate",
                                     lambda m: self.set_pose(m.position[0], m.position[1], m.imu_state.rpy[2]),
                                     qos_profile_sensor_data)
        elif odom == "odometry":
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry, odom_topic or "/odom", self.on_odometry, qos_profile_sensor_data)

        self.state, self.since, self.busy_until = IDLE, 0.0, 0.0
        self.vel, self.last_sent, self.keys = (0.0, 0.0), 0.0, 0
        self.start, self.out_dist, self.back_dist = None, 0.0, 0.0
        self.create_timer(0.02, self.tick)
        self.get_logger().info(
            "ready%s: trigger %s or /reflex/dash -> %s %.2f m/s for %.2f s (%.2f m), %s; odometry: %s" % (
                " (DRY RUN)" if self.dry_run else "",
                ("button %s" % self.button.upper()) if self.button != "none" else "no button",
                self.direction, self.speed, self.duration, self.speed * self.duration,
                ("back at %.2f m/s after %.1f s" % (self.return_speed, self.hold)) if self.come_back else "no way back",
                odom))

    # --- inputs -------------------------------------------------------------------
    def set_pose(self, x, y, yaw):
        self.pose, self.pose_time = (float(x), float(y), float(yaw)), time.monotonic()

    def on_odometry(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.set_pose(m.pose.pose.position.x, m.pose.pose.position.y, yaw)

    def on_remote(self, m):
        bit = 1 << BUTTONS[self.button]
        pressed, was = bool(m.keys & bit), bool(self.keys & bit)
        self.keys = m.keys
        if pressed and not was:
            self.on_dash("button " + self.button.upper())

    def odom_fresh(self):
        return self.pose is not None and time.monotonic() - self.pose_time < ODOM_TIMEOUT

    def away(self):
        """Metres moved in the dodge direction since the trigger (odometry), or None."""
        if self.start is None or not self.odom_fresh():
            return None
        x0, y0, yaw = self.start
        dx, dy = self.pose[0] - x0, self.pose[1] - y0
        fx, fy = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy  # body frame
        ux, uy = DIRS[self.direction]
        return fx * ux + fy * uy

    # --- output -------------------------------------------------------------------
    def send(self, api_id, parameter=""):
        if self.dry_run:
            self.get_logger().info("dry run: api_id %d %s" % (api_id, parameter))
            return
        req = Request()
        req.header.identity.id = time.monotonic_ns()
        req.header.identity.api_id = api_id
        req.header.policy.noreply = api_id == MOVE
        req.parameter = parameter
        self.pub.publish(req)

    def move(self, vx, vy):
        self.vel = (vx, vy)
        if vx or vy:
            self.send(MOVE, json.dumps({"x": round(vx, 3), "y": round(vy, 3), "z": 0.0}))
        else:
            self.send(STOP_MOVE, "{}")
        self.last_sent = time.monotonic()

    def event(self, name, **data):
        a = self.away()
        if a is not None:
            data["away_m"] = round(a, 3)
        data.update(event=name, wall_time=time.time())
        self.event_pub.publish(String(data=json.dumps(data)))
        self.get_logger().info("%s %s" % (name, json.dumps({k: v for k, v in data.items() if k != "wall_time"})))

    def enter(self, state, **data):
        self.state, self.since = state, time.monotonic()
        self.event(state, **data)

    # --- behaviour ------------------------------------------------------------------
    def on_dash(self, source):
        now = time.monotonic()
        if self.state != IDLE or now < self.busy_until:
            return self.event("ignored", source=source, state=self.state)
        self.start = self.pose if self.odom_fresh() else None
        ux, uy = DIRS[self.direction]
        self.enter(OUT, source=source, speed=self.speed, duration=self.duration, odometry=self.start is not None)
        self.move(ux * self.speed, uy * self.speed)

    def stop(self, reason):
        self.move(0.0, 0.0)
        self.busy_until = time.monotonic() + self.refractory
        if self.state != IDLE:
            self.enter(IDLE, reason=reason)
        else:
            self.event("stop", reason=reason)

    def finish(self, reason):
        self.move(0.0, 0.0)
        self.busy_until = time.monotonic() + self.refractory
        self.enter(IDLE, reason=reason, out_m=round(self.out_dist, 3))

    def tick(self):
        if self.state == IDLE:
            return
        now = time.monotonic()
        t = now - self.since
        ux, uy = DIRS[self.direction]
        if self.state == OUT:
            if t >= self.duration:
                self.move(0.0, 0.0)
                a = self.away()
                self.out_dist = a if a is not None else self.speed * self.duration
                self.enter(HOLD, out_m=round(self.out_dist, 3))
                if not self.come_back:
                    self.finish("no way back")
        elif self.state == HOLD:
            if t >= self.hold:
                a = self.away()
                if a is not None:
                    self.out_dist, self.back_dist = a, a              # walk back exactly what was measured
                else:
                    self.back_dist = self.out_dist * self.return_scale  # no odometry: a bit short, never past start
                if self.back_dist <= self.tolerance:
                    return self.finish("already at the start")
                self.enter(RETURN, to_go_m=round(self.back_dist, 3), odometry=a is not None)
                self.move(-ux * self.return_speed, -uy * self.return_speed)
        elif self.state == RETURN:
            a = self.away()
            limit = 2.0 * self.back_dist / self.return_speed + 1.0
            if a is not None and self.start is not None:
                if a <= self.tolerance:
                    return self.finish("back at the start")
            elif t >= self.back_dist / self.return_speed:
                return self.finish("walked back by time")
            if t >= limit:
                return self.finish("time limit")
        if (self.vel[0] or self.vel[1]) and now - self.last_sent > 0.1:
            self.move(*self.vel)


def main():
    rclpy.init()
    node = Go2Dodge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            node.send(STOP_MOVE, "{}")
            time.sleep(0.05)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
