"""Isolated check of Nav2 Collision Monitor behaviour for one config: a static wall of points beside the
robot, a constant sideways velocity request, and what comes out.

  python cm_probe.py <params.yaml> [wall_y] [vy]
Prints the output velocity for each request. Uses the private test domain (see avoid_sim.ros_env).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avoid_sim import ros_env  # noqa: E402


def main() -> None:
    params = sys.argv[1]
    wall_y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    vy = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    import os
    os.environ.update(ros_env())
    log = open("/tmp/cm_probe.log", "w")
    cm = subprocess.Popen(["ros2", "run", "nav2_collision_monitor", "collision_monitor", "--ros-args",
                           "--params-file", params], stdout=log, stderr=subprocess.STDOUT, env=ros_env())
    import rclpy
    from geometry_msgs.msg import Twist
    from lifecycle_msgs.srv import ChangeState
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    rclpy.init()
    n = rclpy.create_node("cm_probe")
    pc = n.create_publisher(PointCloud2, "/avoid/cloud_obstacles", 10)
    ck = n.create_publisher(Clock, "/clock", 10)
    cmd = n.create_publisher(Twist, "/cmd_vel_raw", 10)
    from geometry_msgs.msg import Point32, PolygonStamped
    fp_pub = n.create_publisher(PolygonStamped, "/avoid/footprint", 10)
    outs = []
    n.create_subscription(Twist, "/cmd_vel", lambda m: outs.append((m.linear.x, m.linear.y, m.angular.z)), 10)

    T = [100.0]

    def stamp(h_or_c, t):
        h_or_c.sec, h_or_c.nanosec = int(t), int((t % 1) * 1e9)

    def tick(dt=0.02):
        T[0] += dt
        c = Clock()
        stamp(c.clock, T[0])
        ck.publish(c)

    cli = n.create_client(ChangeState, "/collision_monitor/change_state")
    while not cli.wait_for_service(timeout_sec=0.5):
        tick()
    for tid in (1, 3):
        req = ChangeState.Request()
        req.transition.id = tid
        f = cli.call_async(req)
        while not f.done():
            tick()
            rclpy.spin_once(n, timeout_sec=0.05)
        print(f"transition {tid}: {f.result().success}")
    xs = np.linspace(-0.4, 0.4, 17)
    zs = np.linspace(0.1, 0.4, 4)
    wall = np.array([[x, wall_y, z] for x in xs for z in zs], np.float32)
    results = []
    for k in range(40):
        tick()
        h = Header()
        h.frame_id = "base_footprint"
        stamp(h.stamp, T[0] - 0.03)
        pc.publish(point_cloud2.create_cloud_xyz32(h, wall.tolist()))
        fp = PolygonStamped()
        fp.header = h
        fp.polygon.points = [Point32(x=x, y=y, z=0.0) for x, y in ((0.47, 0.25), (0.47, -0.25), (-0.47, -0.25), (-0.47, 0.25))]
        fp_pub.publish(fp)
        time.sleep(0.02)
        tw = Twist()
        tw.linear.y = vy
        n_before = len(outs)
        cmd.publish(tw)
        end = time.time() + 0.3
        while len(outs) == n_before and time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.005)
        results.append(outs[-1] if len(outs) > n_before else None)
    print(f"wall at y={wall_y} m, request vy={vy}: last 5 outputs {results[-5:]}")
    cm.terminate()
    cm.wait(5)
    n.destroy_node()
    rclpy.shutdown()
    log.close()
    print(open("/tmp/cm_probe.log").read()[-1500:])


if __name__ == "__main__":
    main()
