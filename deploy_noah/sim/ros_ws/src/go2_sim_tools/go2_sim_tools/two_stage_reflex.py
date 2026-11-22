"""Two-stage dodge reflex: lean in when armed, dash back on the throw, return to the start otherwise.

Input  /reflex/arm         std_msgs/Empty: someone winds up (stage 1)
       /reflex/dash        std_msgs/Empty: the ball was thrown (stage 2)
       /reflex/disarm      std_msgs/Empty: the wind-up was aborted
       /durance/clearance  LiDAR safety state from obstacle_detection
       odom_topic          robot position (nav_msgs/Odometry); without it distances are
                           dead-reckoned from the commanded speed
Output /api/sport/request  Unitree sport API: Move / StopMove
       /reflex/event       JSON event log (every state change, ignored triggers) for timing

arm:    only accepted at the start position (state idle). The robot walks forward at
        `approach_speed` for at most `approach_dist`, so it is already stepping when the
        decision comes, and stays armed for `arm_time`.
dash:   only accepted while armed. The robot backs off at `retreat_speed` until it is
        `retreat_dist` behind the point where the dash came, waits `hold_time` and walks
        back to the start.
disarm, or `arm_time` running out: the robot walks back to the start.
The triggers come from gesture_trigger (SNN scores) or reflex_keyboard (keys, for tests).
Safe defaults: forward motion only with a CLEAR lidar state, every motion has a time limit.
"""

import json
import math
import time

import rclpy
from msgs.msg import Clearance
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Empty, String
from unitree_api.msg import Request

MOVE, STOP_MOVE = 1008, 1003
IDLE, ARMED, DASH, HOLD, RETURN = "idle", "armed", "dash", "hold", "return"
TICK = 0.02
RESEND = 0.1  # the Go2 drops a velocity that is not refreshed


class TwoStageReflex(Node):
    def __init__(self):
        super().__init__("two_stage_reflex")
        p = self.declare_parameter
        self.arm_time = p("arm_time", 1.0).value
        self.approach_speed = p("approach_speed", 0.3).value  # 0: stay put while armed
        self.approach_dist = p("approach_dist", 0.15).value
        self.retreat_speed = p("retreat_speed", 1.0).value
        self.retreat_dist = p("retreat_dist", 0.5).value
        self.return_speed = p("return_speed", 0.3).value
        self.hold_time = p("hold_time", 1.0).value
        self.tolerance = p("tolerance", 0.03).value
        odom_topic = p("odom_topic", "/odom/ground_truth").value

        self.req_pub = self.create_publisher(Request, "/api/sport/request", 10)
        self.event_pub = self.create_publisher(String, "/reflex/event", 10)
        self.create_subscription(Empty, "/reflex/arm", self.on_arm, 10)
        self.create_subscription(Empty, "/reflex/dash", self.on_dash, 10)
        self.create_subscription(Empty, "/reflex/disarm", self.on_disarm, 10)
        self.create_subscription(Clearance, "/durance/clearance", self.on_clearance, 1)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 1)
        self.create_timer(TICK, self.tick)

        self.clearance = Clearance.UNKNOWN
        self.state = IDLE
        self.since = time.monotonic()   # start of the current state
        self.pose = None                # (x, y, yaw) from odometry
        self.origin = None              # pose when armed
        self.reckoned = 0.0             # distance along the heading without odometry
        self.target = 0.0               # distance at which the current motion ends
        self.speed = 0.0                # commanded forward speed
        self.sent = 0.0
        self.request_id = 0
        self.trial = 0
        self.get_logger().info(
            f"two_stage_reflex ready: armed for {self.arm_time} s, approach {self.approach_speed} m/s up to "
            f"{self.approach_dist} m, retreat {self.retreat_speed} m/s for {self.retreat_dist} m")

    # --- helpers -------------------------------------------------------------
    def send(self, api_id, parameter=None):
        self.request_id += 1
        req = Request()
        req.header.identity.id = self.request_id
        req.header.identity.api_id = api_id
        req.parameter = json.dumps(parameter) if parameter is not None else ""
        self.req_pub.publish(req)

    def drive(self, speed):
        """Set the forward speed; 0 stops."""
        self.speed = speed
        self.sent = time.monotonic()
        if speed:
            self.send(MOVE, {"x": speed, "y": 0.0, "z": 0.0})
        else:
            self.send(STOP_MOVE)

    def distance(self):
        """Metres moved forward since the robot was armed, along the heading it had then."""
        if self.pose is None or self.origin is None:
            return self.reckoned
        x0, y0, yaw = self.origin
        return (self.pose[0] - x0) * math.cos(yaw) + (self.pose[1] - y0) * math.sin(yaw)

    def event(self, name, **data):
        data.update(event=name, trial=self.trial, wall_time_ns=time.time_ns(), distance=round(self.distance(), 3))
        self.event_pub.publish(String(data=json.dumps(data)))
        self.get_logger().info(f"{name}: {data}")

    def enter(self, state, **data):
        self.state = state
        self.since = time.monotonic()
        self.event(state, **data)

    # --- inputs --------------------------------------------------------------
    def on_clearance(self, msg):
        self.clearance = msg.state

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)

    def on_arm(self, _msg):
        if self.state != IDLE:
            self.event("ignored", trigger="arm", state=self.state)  # not back at the start yet
            return
        self.trial += 1
        self.origin, self.reckoned = self.pose, 0.0
        self.target = self.approach_dist
        self.enter(ARMED)
        if self.approach_speed > 0 and self.clearance == Clearance.CLEAR:
            self.drive(self.approach_speed)

    def on_dash(self, _msg):
        if self.state != ARMED:
            self.event("ignored", trigger="dash", state=self.state)
            return
        self.target = self.distance() - self.retreat_dist
        self.enter(DASH)
        self.drive(-self.retreat_speed)

    def on_disarm(self, _msg):
        if self.state == ARMED:
            self.go_home("disarm")

    # --- behaviour -----------------------------------------------------------
    def go_home(self, reason):
        self.target = 0.0
        self.enter(RETURN, reason=reason)
        d = self.distance()
        self.drive(math.copysign(self.return_speed, -d) if abs(d) > self.tolerance else 0.0)

    def tick(self):
        now = time.monotonic()
        self.reckoned += self.speed * TICK
        d, elapsed = self.distance(), now - self.since

        if self.state == ARMED:
            if self.speed and d >= self.target:
                self.drive(0.0)                      # leaned in far enough, wait for the decision
            if elapsed > self.arm_time:
                self.go_home("timeout")
        elif self.state == DASH:
            if d <= self.target or elapsed > 2 * self.retreat_dist / self.retreat_speed + 1.0:
                self.drive(0.0)
                self.enter(HOLD)
        elif self.state == HOLD:
            if elapsed > self.hold_time:
                self.go_home("after dash")
        elif self.state == RETURN:
            done = d <= self.target if self.speed < 0 else d >= self.target
            if not self.speed or done or elapsed > 2 * (self.retreat_dist + self.approach_dist) / self.return_speed + 1.0:
                self.drive(0.0)
                self.enter(IDLE)
        if self.speed and now - self.sent > RESEND:
            self.drive(self.speed)


def main():
    rclpy.init()
    node = TwoStageReflex()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
