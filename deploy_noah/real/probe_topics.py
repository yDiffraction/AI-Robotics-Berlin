"""Read-only: how often the topics Noah's reflex depends on arrive on this laptop (sends nothing).

  /utlidar/robot_pose, /utlidar/height_map_array   from the Go2 (obstacle_detection's input)
  /utlidar/robot_odom                              from the Go2 (two_stage_reflex's distances)
  /api/sport/response                              the Go2's sport service answering anyone
  /durance/gestures, /durance/clearance            only while real_robot.launch.py runs (dry or robot)

  run_noah.ps1 topics            (or: source real/env.sh robot && python3 real/probe_topics.py [seconds])
"""
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from msgs.msg import Clearance, Gestures
from rclpy.qos import QoSProfile, ReliabilityPolicy
from unitree_api.msg import Response
from unitree_go.msg import HeightMap

TOPICS = [("/utlidar/robot_pose", PoseStamped, True), ("/utlidar/robot_odom", Odometry, True), ("/utlidar/height_map_array", HeightMap, True),
          ("/api/sport/response", Response, False), ("/durance/gestures", Gestures, False),
          ("/durance/clearance", Clearance, False)]


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    rclpy.init()
    node = rclpy.create_node("noah_probe_topics")
    count, last = {}, {}
    for name, typ, best_effort in TOPICS:
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE)

        def cb(msg, name=name):
            count[name] = count.get(name, 0) + 1
            last[name] = msg
        node.create_subscription(typ, name, cb, qos)
    print(f"listening {seconds:.0f} s ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
    for name, _, _ in TOPICS:
        n = count.get(name, 0)
        extra = ""
        if name == "/utlidar/height_map_array" and n:
            m = last[name]
            extra = f"  ({m.width}x{m.height} cells, {m.resolution} m)"
        if name == "/durance/clearance" and n:
            extra = f"  (last state {['CLEAR', 'OBSTACLE', 'UNKNOWN'][last[name].state]})"
        print(f"{name:30s} {n / seconds:6.1f} Hz{extra}")
    if not count.get("/utlidar/height_map_array"):
        print("no height map: obstacle_detection will never report CLEAR, so throw_reflex can only duck. "
              "If the pose arrives but the map does not, large DDS messages are lost on the way (see README).")
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
