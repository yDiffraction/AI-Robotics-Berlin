"""Trigger two_stage_reflex from the keyboard, for tests without the camera.

  a  arm     the robot leans in and stays armed for a moment (arm_time of two_stage_reflex)
  d  dash    while armed: the robot backs off
  s  disarm  while armed: the robot walks back to the start
  q  quit

Arming again only works once the robot is back at the start. Needs a terminal:
  ros2 run go2_sim_tools reflex_keyboard        (./sim.sh keys)
"""

import json
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

KEYS = {"a": "arm", "d": "dash", "s": "disarm"}
TEXT = {"armed": "ARMED - press d to dash", "dash": "DASH", "hold": "dodged, waiting",
        "return": "walking back to the start", "idle": "at the start - press a to arm"}


class ReflexKeyboard(Node):
    def __init__(self):
        super().__init__("reflex_keyboard")
        self.pubs = {name: self.create_publisher(Empty, f"/reflex/{name}", 10) for name in KEYS.values()}
        self.create_subscription(String, "/reflex/event", self.on_event, 10)

    def on_event(self, msg):
        e = json.loads(msg.data)
        if e["event"] == "ignored":
            print(f"  {e['trigger']} ignored: the robot is in state '{e['state']}'")
        else:
            reason = f" ({e['reason']})" if "reason" in e else ""
            print(f"  {TEXT.get(e['event'], e['event'])}{reason}   [{e['distance'] * 100:+.0f} cm]")


def main():
    if not sys.stdin.isatty():
        raise SystemExit("reflex_keyboard needs a terminal (ros2 run, not ros2 launch)")
    rclpy.init()
    node = ReflexKeyboard()
    print(__doc__.split("\n\n")[1])
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "q":
                    break
                if key in KEYS:
                    node.pubs[KEYS[key]].publish(Empty())
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.try_shutdown()
