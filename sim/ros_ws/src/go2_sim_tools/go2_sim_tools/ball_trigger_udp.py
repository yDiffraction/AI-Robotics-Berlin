"""UDP -> ROS bridge for the ball trigger: tools/ball_live.py (laptop: camera -> binary frames -> SNN on the
Kria -> BallTrigger) sends one JSON datagram per event, {"event": "arm"|"dash"|"disarm", "t": ..., "frame": ...};
this node republishes it as std_msgs/Empty on /reflex/arm, /reflex/dash, /reflex/disarm for two_stage_reflex,
and the raw JSON (plus receive time) on /reflex/trigger_log.

Safety: datagrams older than `max_age` s (host_time in the message vs. this machine's clock, only checked when
`check_age` is true, i.e. clocks are synced) are dropped; unknown events are ignored.

  ros2 run go2_sim_tools ball_trigger_udp --ros-args -p port:=5555
"""

import json
import socket
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

EVENTS = ("arm", "dash", "disarm")


class BallTriggerUdp(Node):
    def __init__(self):
        super().__init__("ball_trigger_udp")
        self.port = self.declare_parameter("port", 5555).value
        self.max_age = self.declare_parameter("max_age", 0.2).value
        self.check_age = self.declare_parameter("check_age", False).value
        self.pubs = {e: self.create_publisher(Empty, "/reflex/" + e, 10) for e in EVENTS}
        self.log_pub = self.create_publisher(String, "/reflex/trigger_log", 10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.setblocking(False)
        self.create_timer(0.002, self.poll)
        self.get_logger().info("listening for ball trigger events on udp/%d" % self.port)

    def poll(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            try:
                msg = json.loads(data.decode())
                ev = msg["event"]
            except (ValueError, KeyError, UnicodeDecodeError):
                self.get_logger().warning("bad datagram from %s" % (addr,))
                continue
            if ev not in EVENTS:
                continue
            age = time.time() - float(msg.get("host_time", time.time()))
            if self.check_age and age > self.max_age:
                self.get_logger().warning("dropped stale %s (%.0f ms old)" % (ev, age * 1e3))
                continue
            self.pubs[ev].publish(Empty())
            msg["rx_time"] = time.time()
            self.log_pub.publish(String(data=json.dumps(msg)))
            self.get_logger().info("%s (frame %s)" % (ev, msg.get("frame")))


def main():
    rclpy.init()
    node = BallTriggerUdp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
