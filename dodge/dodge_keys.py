#!/usr/bin/env python3
"""Keyboard trigger for go2_dodge.py:  d / Enter = dodge (/reflex/dash),  space = STOP (/dodge/stop),  q = quit.

  source /opt/ros/humble/setup.bash && python3 dodge_keys.py
Without a terminal (e.g. from an agent) use: ros2 topic pub --once /reflex/dash std_msgs/msg/Empty "{}"
"""

import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String


class DodgeKeys(Node):
    def __init__(self):
        super().__init__("dodge_keys")
        self.dash = self.create_publisher(Empty, "/reflex/dash", 10)
        self.stop = self.create_publisher(Empty, "/dodge/stop", 10)
        self.create_subscription(String, "/dodge/event", lambda m: print("  robot:", m.data), 10)


def main():
    rclpy.init()
    node = DodgeKeys()
    settings = termios.tcgetattr(sys.stdin)
    print(__doc__.split("\n")[0])
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()
                if key in ("d", "\n", "\r"):
                    node.dash.publish(Empty()); print("dash")
                elif key == " ":
                    node.stop.publish(Empty()); print("STOP")
                elif key in ("q", ""):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop.publish(Empty())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
