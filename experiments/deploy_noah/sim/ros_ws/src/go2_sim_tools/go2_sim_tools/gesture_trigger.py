"""Turn the SNN's class scores into the triggers of two_stage_reflex.

Input  /durance/gestures  class scores from the SNN (via gesture_node)
Output /reflex/arm        the share of all non-idle classes exceeds `windup_threshold`
       /reflex/dash       the throw share exceeds `threshold` while armed
       /reflex/disarm     the feint share exceeds `threshold`, or the scores go idle again

Every condition has to hold for `confirm_frames` frames. After a dash or disarm the scores
have to go idle once before the next arm, so one gesture triggers only once.
"""

import rclpy
from msgs.msg import Gestures
from rclpy.node import Node
from std_msgs.msg import Empty


class GestureTrigger(Node):
    def __init__(self):
        super().__init__("gesture_trigger")
        p = self.declare_parameter
        self.throw_class = p("throw_class", 0).value
        self.feint_class = p("feint_class", 1).value      # -1: the model has no feint class
        self.idle_class = p("idle_class", 2).value
        self.windup_threshold = p("windup_threshold", 0.5).value
        self.threshold = p("threshold", 0.6).value
        self.confirm_frames = p("confirm_frames", 2).value
        self.pubs = {name: self.create_publisher(Empty, f"/reflex/{name}", 10) for name in ("arm", "dash", "disarm")}
        self.create_subscription(Gestures, "/durance/gestures", self.on_gestures, 1)
        self.streak = {"active": 0, "throw": 0, "feint": 0, "idle": 0}
        self.armed = False
        self.ready = True   # the scores were idle since the last gesture

    def on_gestures(self, msg):
        total = sum(msg.scores) or 1.0
        p_throw = msg.scores[self.throw_class] / total
        p_feint = msg.scores[self.feint_class] / total if self.feint_class >= 0 else 0.0
        p_active = 1.0 - msg.scores[self.idle_class] / total
        holds = {"active": p_active > self.windup_threshold, "throw": p_throw > self.threshold,
                 "feint": p_feint > self.threshold, "idle": p_active < self.windup_threshold}
        for name, hold in holds.items():
            self.streak[name] = self.streak[name] + 1 if hold else 0
        seen = {name for name, n in self.streak.items() if n >= self.confirm_frames}

        if "idle" in seen:
            self.ready = True
        if self.armed and "throw" in seen:
            self.trigger("dash", f"throw {p_throw:.2f}")
        elif self.armed and seen & {"feint", "idle"}:
            self.trigger("disarm", "feint" if "feint" in seen else "idle again")
        elif not self.armed and self.ready and "active" in seen:
            self.trigger("arm", f"active {p_active:.2f}")

    def trigger(self, name, why):
        self.armed = name == "arm"
        self.ready = False
        self.pubs[name].publish(Empty())
        self.get_logger().info(f"{name} ({why})")


def main():
    rclpy.init()
    node = GestureTrigger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
