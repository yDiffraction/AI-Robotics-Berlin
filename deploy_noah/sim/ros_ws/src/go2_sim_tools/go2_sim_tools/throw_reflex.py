"""Dodge reflex: sidestep a completed throw, ignore a feint.

Input  /durance/gestures   class scores from the SNN (via gesture_node)
       /durance/clearance  LiDAR safety state from obstacle_detection
Output /api/sport/request  Unitree sport API: Move / StopMove / StandDown / StandUp
       /reflex/event       JSON event log (detections, actions, timing) for evaluation

Decision: the throw probability (score share of `throw_class`) must exceed `threshold`
for `confirm_frames` consecutive frames. If the space is clear the robot sidesteps
(Move with lateral speed), otherwise it ducks (StandDown) since a sidestep could hit
something. Feints are logged and ignored. Safe defaults: one action at a time with a
cooldown, and any motion is stopped if the score stream goes stale.
"""

import json
import time

import rclpy
from msgs.msg import Clearance, Gestures
from rclpy.node import Node
from std_msgs.msg import String
from unitree_api.msg import Request

MOVE, STOP_MOVE, STAND_DOWN, STAND_UP = 1008, 1003, 1005, 1004


class ThrowReflex(Node):
    def __init__(self):
        super().__init__("throw_reflex")
        self.throw_class = self.declare_parameter("throw_class", 0).value
        self.feint_class = self.declare_parameter("feint_class", 1).value
        self.threshold = self.declare_parameter("threshold", 0.6).value
        self.confirm_frames = self.declare_parameter("confirm_frames", 2).value
        self.dodge_speed = self.declare_parameter("dodge_speed", 0.8).value
        self.dodge_time = self.declare_parameter("dodge_time", 1.0).value
        self.alternate = self.declare_parameter("alternate_sides", True).value
        self.cooldown = self.declare_parameter("cooldown", 3.0).value
        self.recenter = self.declare_parameter("recenter", True).value
        self.stale_timeout = self.declare_parameter("stale_timeout", 0.5).value

        self.req_pub = self.create_publisher(Request, "/api/sport/request", 10)
        self.event_pub = self.create_publisher(String, "/reflex/event", 10)
        self.create_subscription(Gestures, "/durance/gestures", self.on_gestures, 1)
        self.create_subscription(Clearance, "/durance/clearance", self.on_clearance, 1)
        self.create_timer(0.05, self.watchdog)

        self.clearance = Clearance.UNKNOWN
        self.streak = 0
        self.feint_streak = 0
        self.busy_until = 0.0
        self.moving = False
        self.side = 1.0
        self.last_scores = 0.0
        self.request_id = 0
        self.stats = {"throws": 0, "dodges": 0, "ducks": 0, "feints_ignored": 0}
        self.get_logger().info(f"throw_reflex ready: throw={self.throw_class} feint={self.feint_class} "
                               f"threshold={self.threshold} confirm={self.confirm_frames}")

    # --- helpers -------------------------------------------------------------
    def send(self, api_id, parameter=None):
        self.request_id += 1
        req = Request()
        req.header.identity.id = self.request_id
        req.header.identity.api_id = api_id
        req.parameter = json.dumps(parameter) if parameter is not None else ""
        self.req_pub.publish(req)

    def event(self, kind, **data):
        data.update(event=kind, wall_time_ns=time.time_ns(), **self.stats)
        self.event_pub.publish(String(data=json.dumps(data)))

    def later(self, delay, fn):
        timer = None

        def fire():
            timer.cancel()
            fn()
        timer = self.create_timer(delay, fire)

    # --- inputs --------------------------------------------------------------
    def on_clearance(self, msg):
        self.clearance = msg.state

    def on_gestures(self, msg):
        now = time.monotonic()
        self.last_scores = now
        total = sum(msg.scores) or 1.0
        p_throw = msg.scores[self.throw_class] / total
        p_feint = msg.scores[self.feint_class] / total

        self.streak = self.streak + 1 if p_throw > self.threshold else 0
        self.feint_streak = self.feint_streak + 1 if p_feint > self.threshold else 0

        if now < self.busy_until:
            return
        if self.feint_streak == self.confirm_frames:
            self.stats["feints_ignored"] += 1
            self.get_logger().info(f"feint detected (p={p_feint:.2f}) -> ignored")
            self.event("feint_ignored", p_feint=p_feint)
        if self.streak >= self.confirm_frames:
            self.stats["throws"] += 1
            self.react(p_throw)

    # --- behaviour -----------------------------------------------------------
    def react(self, p_throw):
        t_detect = time.time_ns()
        self.streak = 0
        self.busy_until = time.monotonic() + self.cooldown
        if self.clearance == Clearance.CLEAR:
            self.stats["dodges"] += 1
            side = self.side
            self.send(MOVE, {"x": 0.0, "y": side * self.dodge_speed, "z": 0.0})
            self.moving = True
            self.get_logger().info(f"THROW (p={p_throw:.2f}) -> sidestep {'left' if side > 0 else 'right'}")
            self.event("dodge", p_throw=p_throw, side=side, t_detect_ns=t_detect, t_command_ns=time.time_ns())
            self.later(self.dodge_time, self.stop)
            if self.recenter:
                self.later(self.cooldown - self.dodge_time - 0.2,
                           lambda: self.send(MOVE, {"x": 0.0, "y": -side * self.dodge_speed, "z": 0.0}))
                self.later(self.cooldown - 0.2, self.stop)
            if self.alternate:
                self.side = -self.side
        else:
            self.stats["ducks"] += 1
            self.send(STAND_DOWN)
            self.get_logger().info(f"THROW (p={p_throw:.2f}) but space not clear -> duck")
            self.event("duck", p_throw=p_throw, t_detect_ns=t_detect, t_command_ns=time.time_ns())
            self.later(1.5, lambda: self.send(STAND_UP))

    def stop(self):
        self.send(STOP_MOVE)
        self.moving = False

    def watchdog(self):
        if self.moving and time.monotonic() - self.last_scores > self.stale_timeout:
            self.get_logger().warn("score stream stale -> StopMove")
            self.event("stale_stop")
            self.stop()


def main():
    rclpy.init()
    node = ThrowReflex()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
