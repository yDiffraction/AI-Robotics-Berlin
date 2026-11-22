#!/usr/bin/env python3
"""Dodge only: the Go2 jumps out of the way when a ROS 2 trigger arrives. Nothing else.

Trigger (std_msgs/Empty), from the keyboard node (dodge_keys.py) or from any shell / agent:
  /reflex/dash   dodge now                ros2 topic pub --once /reflex/dash std_msgs/msg/Empty "{}"
  /dodge/stop    stop at once (StopMove)  ros2 topic pub --once /dodge/stop std_msgs/msg/Empty "{}"

Dodge = sport API Move (api_id 1008) at `speed` m/s in `direction` for `duration` s, the velocity re-sent
every 0.1 s (the Go2 drops a velocity that is not refreshed), then StopMove (1003). With `come_back` the
robot waits `hold` s and walks back the same way at `return_speed`. Time based, no odometry needed.
Messages go to /api/sport/request (unitree_api/msg/Request), like the starter's action_bridge.

Safety: speeds and durations are clamped; a new dash is ignored while one runs and for `refractory` s
after it; /dodge/stop and Ctrl+C send StopMove (3 times on exit); `dry_run:=true` only logs.

Run (ROS 2 Humble + the go2-kria overlay that has unitree_api, Go2 on DDS domain 0):
  source /opt/ros/humble/setup.bash && source <overlay>/local_setup.bash
  python3 go2_dodge.py                                   # back off 0.4 m at 0.8 m/s
  python3 go2_dodge.py --ros-args -p direction:=left -p speed:=0.6 -p duration:=0.5
  python3 go2_dodge.py --ros-args -p dry_run:=true       # log only, nothing sent
Status: /dodge/event (std_msgs/String, JSON) for every dash, stop and ignored trigger.
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String
from unitree_api.msg import Request

MOVE, STOP_MOVE = 1008, 1003
DIRS = {"back": (-1.0, 0.0), "forward": (1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}


class Go2Dodge(Node):
    def __init__(self):
        super().__init__("go2_dodge")
        p = self.declare_parameter
        self.direction = p("direction", "back").value       # back | left | right | forward
        self.speed = min(max(p("speed", 0.8).value, 0.1), 1.5)         # m/s
        self.duration = min(max(p("duration", 0.5).value, 0.1), 2.0)    # s of movement
        self.come_back = p("come_back", True).value
        self.hold = min(max(p("hold", 1.0).value, 0.0), 5.0)            # s to wait before coming back
        self.return_speed = min(max(p("return_speed", 0.3).value, 0.1), 0.5)
        self.refractory = p("refractory", 1.0).value         # s after a dodge before the next is accepted
        self.dry_run = p("dry_run", False).value
        if self.direction not in DIRS:
            raise SystemExit("direction must be one of %s" % ", ".join(DIRS))

        self.pub = self.create_publisher(Request, "/api/sport/request", 10)
        self.event_pub = self.create_publisher(String, "/dodge/event", 10)
        for topic in ("/reflex/dash", "/dodge/dash"):
            self.create_subscription(Empty, topic, lambda _m, t=topic: self.on_dash(t), 10)
        for topic in ("/dodge/stop", "/reflex/stop"):
            self.create_subscription(Empty, topic, lambda _m: self.stop("stop topic"), 10)

        self.plan = []          # [(t_end, vx, vy)] segments of the running dodge, vx = vy = 0 means wait
        self.busy_until = 0.0
        self.last_sent = 0.0
        self.create_timer(0.02, self.tick)
        dx, dy = DIRS[self.direction]
        self.get_logger().info(
            "ready%s: dash -> %s %.2f m/s for %.2f s (%.2f m)%s; trigger: ros2 topic pub --once /reflex/dash "
            "std_msgs/msg/Empty \"{}\"" % (" (DRY RUN)" if self.dry_run else "", self.direction, self.speed,
                                          self.duration, self.speed * self.duration,
                                          ", then back" if self.come_back else ""))

    # --- output -----------------------------------------------------------------
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
        if vx or vy:
            self.send(MOVE, json.dumps({"x": vx, "y": vy, "z": 0.0}))
        else:
            self.send(STOP_MOVE, "{}")
        self.last_sent = time.monotonic()

    def event(self, name, **data):
        data.update(event=name, wall_time=time.time())
        self.event_pub.publish(String(data=json.dumps(data)))
        self.get_logger().info("%s %s" % (name, json.dumps({k: v for k, v in data.items() if k != "wall_time"})))

    # --- triggers ---------------------------------------------------------------
    def on_dash(self, topic):
        now = time.monotonic()
        if now < self.busy_until:
            return self.event("ignored", topic=topic, reason="dodge running or refractory",
                              wait=round(self.busy_until - now, 2))
        dx, dy = DIRS[self.direction]
        out_end = now + self.duration
        self.plan = [(out_end, dx * self.speed, dy * self.speed)]
        end = out_end
        if self.come_back:
            back_time = self.speed * self.duration / self.return_speed
            self.plan += [(out_end + self.hold, 0.0, 0.0),
                          (out_end + self.hold + back_time, -dx * self.return_speed, -dy * self.return_speed)]
            end = out_end + self.hold + back_time
        self.busy_until = end + self.refractory
        self.event("dash", topic=topic, direction=self.direction, speed=self.speed, duration=self.duration)
        self.move(self.plan[0][1], self.plan[0][2])

    def stop(self, reason):
        self.plan = []
        self.move(0.0, 0.0)
        self.busy_until = time.monotonic() + self.refractory
        self.event("stop", reason=reason)

    # --- loop -------------------------------------------------------------------
    def tick(self):
        if not self.plan:
            return
        now = time.monotonic()
        changed = False
        while self.plan and now >= self.plan[0][0]:
            self.plan.pop(0)
            changed = True
        if not self.plan:
            self.move(0.0, 0.0)
            self.event("done")
            return
        _, vx, vy = self.plan[0]
        if changed or (vx or vy) and now - self.last_sent > 0.1:
            self.move(vx, vy)


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
