"""Translate Unitree sport API requests into motions of the CHAMP Go2 in Gazebo.

The real Go2 executes /api/sport/request itself. The simulated Go2 is driven by a
CHAMP controller that only understands a velocity (/cmd_vel) and a body pose
(/body_pose), so this node maps each api_id onto a short motion script and answers
on /api/sport/response like the robot would (code 0 = accepted).

  1008 Move        parameter {"x": vx, "y": vy, "z": wz}; held for move_hold seconds
  1003 StopMove    zero velocity
  1009 Sit, 1005 StandDown          lower the body
  1004 StandUp, 1010 RiseSit, 1002  neutral pose
  1016 Hello, 1022 Dance, others    small visible pose animations
"""

import json
import math

import rclpy
from geometry_msgs.msg import Pose, Twist
from rclpy.node import Node
from unitree_api.msg import Request, Response


def quat(roll=0.0, pitch=0.0, yaw=0.0):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class SportApiBridge(Node):
    def __init__(self):
        super().__init__("sport_api_bridge")
        self.declare_parameter("move_hold", 0.8)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pose_pub = self.create_publisher(Pose, "/body_pose", 10)
        self.resp_pub = self.create_publisher(Response, "/api/sport/response", 10)
        self.create_subscription(Request, "/api/sport/request", self.on_request, 10)
        self.twist = Twist()
        self.twist_until = 0.0
        self.anim = None          # (name, start_time, duration)
        self.base = (0.0, 0.0, 0.0, 0.0)  # z offset, roll, pitch, yaw
        self.create_timer(0.02, self.tick)
        self.get_logger().info("bridging /api/sport/request -> /cmd_vel + /body_pose")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_request(self, req):
        api_id = req.header.identity.api_id
        t = self.now()
        if api_id == 1008:
            try:
                p = json.loads(req.parameter or "{}")
            except json.JSONDecodeError:
                p = {}
            self.twist = Twist()
            self.twist.linear.x = float(p.get("x", 0.0))
            self.twist.linear.y = float(p.get("y", 0.0))
            self.twist.angular.z = float(p.get("z", 0.0))
            self.twist_until = t + self.get_parameter("move_hold").value
        elif api_id == 1003:
            self.twist, self.twist_until = Twist(), 0.0
        elif api_id == 1009:
            self.base, self.anim = (-0.08, 0.0, -0.35, 0.0), None
        elif api_id == 1005:
            self.base, self.anim = (-0.14, 0.0, 0.0, 0.0), None
        elif api_id in (1002, 1004, 1006, 1010):
            self.base, self.anim = (0.0, 0.0, 0.0, 0.0), None
        elif api_id == 1022:
            self.anim = ("dance", t, 6.0)
        elif api_id == 1016:
            self.anim = ("hello", t, 2.0)
        else:
            self.anim = ("nod", t, 1.5)
        self.get_logger().info(f"api_id={api_id} parameter={req.parameter!r}")

        resp = Response()
        resp.header.identity.id = req.header.identity.id
        resp.header.identity.api_id = api_id
        resp.header.status.code = 0
        self.resp_pub.publish(resp)

    def tick(self):
        t = self.now()
        self.cmd_pub.publish(self.twist if t < self.twist_until else Twist())

        z, roll, pitch, yaw = self.base
        if self.anim:
            name, start, duration = self.anim
            phase = t - start
            if phase > duration:
                self.anim = None
            elif name == "dance":
                roll += 0.25 * math.sin(2 * math.pi * 1.5 * phase)
                yaw += 0.3 * math.sin(2 * math.pi * 0.75 * phase)
                z += 0.03 * math.sin(2 * math.pi * 3.0 * phase)
            elif name == "hello":
                roll += 0.3 * math.sin(2 * math.pi * 2.0 * phase)
                pitch -= 0.15
            elif name == "nod":
                pitch += 0.2 * math.sin(2 * math.pi * 1.5 * phase)
        pose = Pose()
        pose.position.z = z
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat(roll, pitch, yaw)
        self.pose_pub.publish(pose)


def main():
    rclpy.init()
    node = SportApiBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
