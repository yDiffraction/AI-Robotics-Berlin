"""Publishes the two LiDAR-derived topics the Go2 provides, for desk testing.

/utlidar/robot_pose        geometry_msgs/PoseStamped  (robot fixed at the origin)
/utlidar/height_map_array  unitree_go/HeightMap       (flat floor, optional obstacle)

Move the obstacle at runtime:
  ros2 param set /fake_lidar obstacle_distance 0.45   # inside the 0.5 m safety radius
  ros2 param set /fake_lidar obstacle_distance 0.0    # remove it
"""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from unitree_go.msg import HeightMap


class FakeLidar(Node):
    def __init__(self):
        super().__init__("fake_lidar")
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("size", 4.0)
        self.declare_parameter("resolution", 0.05)
        self.declare_parameter("obstacle_distance", 0.0)
        self.declare_parameter("obstacle_bearing_deg", 0.0)
        self.declare_parameter("obstacle_height", 0.5)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pose_pub = self.create_publisher(PoseStamped, "/utlidar/robot_pose", qos)
        self.map_pub = self.create_publisher(HeightMap, "/utlidar/height_map_array", qos)
        rate = self.get_parameter("rate").value
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info("publishing fake pose and height map")

    def tick(self):
        now = self.get_clock().now()
        pose = PoseStamped()
        pose.header.stamp = now.to_msg()
        pose.header.frame_id = "odom"
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        size = self.get_parameter("size").value
        resolution = self.get_parameter("resolution").value
        cells = int(round(size / resolution))
        grid = HeightMap()
        grid.stamp = now.nanoseconds * 1e-9
        grid.frame_id = "odom"
        grid.resolution = resolution
        grid.width = cells
        grid.height = cells
        grid.origin = [-size / 2.0, -size / 2.0]
        data = [0.0] * (cells * cells)

        distance = self.get_parameter("obstacle_distance").value
        if distance > 0.0:
            bearing = math.radians(self.get_parameter("obstacle_bearing_deg").value)
            ix = int(round((distance * math.cos(bearing) - grid.origin[0]) / resolution))
            iy = int(round((distance * math.sin(bearing) - grid.origin[1]) / resolution))
            height = self.get_parameter("obstacle_height").value
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x, y = ix + dx, iy + dy
                    if 0 <= x < cells and 0 <= y < cells:
                        data[cells * y + x] = height
        grid.data = data
        self.map_pub.publish(grid)


def main():
    rclpy.init()
    node = FakeLidar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
