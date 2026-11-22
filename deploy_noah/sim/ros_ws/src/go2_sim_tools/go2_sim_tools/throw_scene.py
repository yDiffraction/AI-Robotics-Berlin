"""Virtual throw scene: a thrower in front of the Go2 who either throws or feints.

Stands in for camera + SNN during desk testing:
  * acts as the SPLEAT server (same TCP/CBOR protocol), streaming class scores at
    `rate` Hz that follow the scripted scene with a `perception_delay`;
  * publishes RViz markers (thrower, arm, ball, scoreboard) in the `world` frame;
  * judges every real throw as DODGED or HIT using Gazebo ground truth.

Score layout matches throw_reflex: [throw, feint, idle, 0, ...] (11 floats, Gestures.msg).
The scores are scripted, not a model: this tests the behaviour, timing and safety logic.
"""

import math
import random
import socket
import struct
import threading
import time

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

THROW, FEINT, IDLE = 0, 1, 2


def cbor_head(major, value):
    if value < 24:
        return bytes([(major << 5) | value])
    if value < 0x100:
        return bytes([(major << 5) | 24, value])
    if value < 0x10000:
        return bytes([(major << 5) | 25]) + struct.pack(">H", value)
    return bytes([(major << 5) | 26]) + struct.pack(">I", value)


def cbor_text(text):
    data = text.encode()
    return cbor_head(3, len(data)) + data


def encode_frame(frame_id, scores):
    payload = struct.pack(f"<{len(scores)}f", *scores)
    body = (cbor_head(5, 4) + cbor_text("id") + cbor_head(0, frame_id % 0xFFFFFFFF)
            + cbor_text("payload") + cbor_head(2, len(payload)) + payload
            + cbor_text("latency") + cbor_head(0, 0) + cbor_text("error") + cbor_text(""))
    return struct.pack(">I", len(body)) + body


class ThrowScene(Node):
    def __init__(self):
        super().__init__("throw_scene")
        self.port = self.declare_parameter("port", 3335).value
        self.rate = self.declare_parameter("rate", 100.0).value
        self.distance = self.declare_parameter("distance", 4.0).value
        self.ball_speed = self.declare_parameter("ball_speed", 4.0).value
        self.windup = self.declare_parameter("windup", 0.6).value
        self.gap = self.declare_parameter("gap", 4.0).value
        self.perception_delay = self.declare_parameter("perception_delay", 0.06).value
        self.throw_ratio = self.declare_parameter("throw_ratio", 0.6).value
        self.hit_radius = self.declare_parameter("hit_radius", 0.28).value

        self.marker_pub = self.create_publisher(MarkerArray, "/throw_scene/markers", 10)
        self.event_pub = self.create_publisher(String, "/throw_scene/event", 10)
        self.create_subscription(Odometry, "/odom/ground_truth", self.on_truth, 10)
        self.robot = None
        self.origin = None
        self.lock = threading.Lock()
        self.kind = "idle"
        self.t0 = time.monotonic()
        self.ball = None   # dict(start, target, t_release, flight, judged)
        self.score = {"throws": 0, "dodged": 0, "hit": 0, "feints": 0}
        self.banner = ""
        self.create_timer(1.0 / 30.0, self.tick)
        threading.Thread(target=self.serve, daemon=True).start()
        self.get_logger().info(f"throw scene: SPLEAT stand-in on port {self.port}")

    def on_truth(self, msg):
        p = msg.pose.pose.position
        self.robot = (p.x, p.y, p.z)
        if self.origin is None:
            self.origin = (p.x, p.y)

    # --- scene timeline -------------------------------------------------------
    def phase(self, now):
        """Return (kind, time since start of this act)."""
        return self.kind, now - self.t0

    def next_act(self, now):
        self.kind = "throw" if random.random() < self.throw_ratio else "feint"
        self.t0 = now
        self.banner = ""

    def scores(self, now):
        kind, t = self.phase(now - self.perception_delay)
        s = [0.0] * 11
        if kind in ("throw", "feint") and 0.0 <= t < self.windup:
            s[THROW], s[FEINT], s[IDLE] = 0.35, 0.35, 0.3      # wind-up looks identical
        elif kind == "throw" and self.windup <= t < self.windup + 0.4:
            s[THROW], s[FEINT], s[IDLE] = 0.85, 0.1, 0.05
        elif kind == "feint" and self.windup <= t < self.windup + 0.4:
            s[THROW], s[FEINT], s[IDLE] = 0.1, 0.85, 0.05
        else:
            s[THROW], s[FEINT], s[IDLE] = 0.05, 0.05, 0.9
        noise = [random.uniform(0, 0.08) for _ in range(3)]
        for i in range(3):
            s[i] += noise[i]
        total = sum(s)
        return [v / total for v in s]

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(1)
        frame_id = 0
        while rclpy.ok():
            conn, _ = server.accept()
            period, nxt = 1.0 / self.rate, time.monotonic()
            try:
                while rclpy.ok():
                    conn.sendall(encode_frame(frame_id, self.scores(time.monotonic())))
                    frame_id += 1
                    nxt += period
                    time.sleep(max(0.0, nxt - time.monotonic()))
            except OSError:
                conn.close()

    # --- animation + judging --------------------------------------------------
    def tick(self):
        if self.robot is None:
            return
        now = time.monotonic()
        kind, t = self.phase(now)
        ox, oy = self.origin
        thrower = (ox + self.distance, oy)
        hand_rest = (thrower[0], thrower[1] + 0.25, 1.35)

        if kind == "idle" and t > self.gap:
            self.next_act(now)
            kind, t = self.phase(now)
        if kind in ("throw", "feint") and t > self.windup + 1.6:
            self.kind, self.t0 = "idle", now

        # arm angle: wind back during wind-up, swing forward on release (throw) or freeze (feint)
        arm = 0.0
        if kind in ("throw", "feint") and t < self.windup:
            arm = -1.6 * (t / self.windup)
        elif kind == "throw" and t < self.windup + 0.25:
            arm = -1.6 + 3.0 * ((t - self.windup) / 0.25)
        elif kind == "throw" and t < self.windup + 1.0:
            arm = 1.4
        elif kind == "feint" and t < self.windup + 0.6:
            arm = -1.6 + 0.6 * ((t - self.windup) / 0.6)
        shoulder = (thrower[0], thrower[1] + 0.25, 1.4)
        hand = (shoulder[0] + 0.6 * math.sin(arm), shoulder[1], shoulder[2] + 0.6 * math.cos(arm))

        if kind == "throw" and self.ball is None and t >= self.windup + 0.15:
            target = (self.robot[0], self.robot[1], 0.3)
            dist = math.dist(hand[:2], target[:2])
            self.ball = {"start": hand, "target": target, "t": now, "flight": dist / self.ball_speed, "judged": False}
            self.score["throws"] += 1
            self.banner = "THROW!"
            self.publish_event("release", flight_s=round(dist / self.ball_speed, 3))
        if kind == "feint" and t >= self.windup and not self.banner:
            self.score["feints"] += 1
            self.banner = "feint..."
            self.publish_event("feint")

        ball_pos = None
        if self.ball:
            b = self.ball
            u = (now - b["t"]) / b["flight"]
            if u <= 1.6:
                sx, sy, sz = b["start"]
                tx, ty, tz = b["target"]
                ball_pos = (sx + (tx - sx) * u, sy + (ty - sy) * u,
                            max(0.05, sz + (tz - sz) * u + 0.6 * u * (1 - u)))
            if u >= 1.0 and not b["judged"]:
                b["judged"] = True
                miss = math.hypot(self.robot[0] - b["target"][0], self.robot[1] - b["target"][1])
                if miss > self.hit_radius:
                    self.score["dodged"] += 1
                    self.banner = f"DODGED ({miss:.2f} m)"
                else:
                    self.score["hit"] += 1
                    self.banner = f"HIT ({miss:.2f} m)"
                self.publish_event("judged", result=self.banner, miss_m=round(miss, 3))
            if u > 1.6:
                self.ball = None

        self.publish_markers(thrower, shoulder, hand, ball_pos)

    def publish_event(self, kind, **data):
        import json
        data.update(event=kind, wall_time_ns=time.time_ns(), **self.score)
        self.event_pub.publish(String(data=json.dumps(data)))
        self.get_logger().info(f"{kind}: {data}")

    def marker(self, mid, mtype, pos, scale, color, text=""):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "throw_scene", mid, mtype, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.text = text
        return m

    def publish_markers(self, thrower, shoulder, hand, ball_pos):
        arr = MarkerArray()
        arr.markers.append(self.marker(0, Marker.CYLINDER, (thrower[0], thrower[1], 0.75), (0.35, 0.35, 1.5),
                                       (0.2, 0.5, 0.9, 1.0)))
        arr.markers.append(self.marker(1, Marker.SPHERE, (thrower[0], thrower[1], 1.65), (0.3, 0.3, 0.3),
                                       (0.95, 0.8, 0.6, 1.0)))
        arm = self.marker(2, Marker.LINE_LIST, (0.0, 0.0, 0.0), (0.08, 0.0, 0.0), (0.95, 0.8, 0.6, 1.0))
        arm.points = [Point(x=shoulder[0], y=shoulder[1], z=shoulder[2]), Point(x=hand[0], y=hand[1], z=hand[2])]
        arr.markers.append(arm)
        held = ball_pos is None and self.kind != "idle" and not (self.ball and self.ball["judged"])
        pos = ball_pos if ball_pos else hand
        ball = self.marker(3, Marker.SPHERE, pos, (0.12, 0.12, 0.12), (1.0, 0.45, 0.0, 1.0 if (ball_pos or held) else 0.0))
        arr.markers.append(ball)
        s = self.score
        board = (f"{self.banner}\nthrows {s['throws']}  dodged {s['dodged']}  hit {s['hit']}\nfeints {s['feints']}")
        arr.markers.append(self.marker(4, Marker.TEXT_VIEW_FACING, (thrower[0], thrower[1], 2.3), (0.0, 0.0, 0.25),
                                       (1.0, 1.0, 1.0, 1.0), board))
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = ThrowScene()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
