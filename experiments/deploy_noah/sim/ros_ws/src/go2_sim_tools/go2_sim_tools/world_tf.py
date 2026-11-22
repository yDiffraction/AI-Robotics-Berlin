"""Anchor the TF tree to Gazebo's ground truth so RViz shows the robot's real motion.

CHAMP's odometry estimate (odom -> base_link) stays near zero in this setup, so this
node publishes world -> odom = ground_truth * inverse(odom -> base_link). With RViz's
fixed frame set to `world`, the model then moves exactly as it does in Gazebo.
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import Buffer, TransformBroadcaster, TransformListener


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def q_inv(q):
    return (-q[0], -q[1], -q[2], q[3])


def q_rot(q, v):
    x, y, z, _ = q_mul(q_mul(q, (v[0], v[1], v[2], 0.0)), q_inv(q))
    return (x, y, z)


class WorldTf(Node):
    def __init__(self):
        super().__init__("world_tf")
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/odom/ground_truth", self.on_truth, 10)

    def on_truth(self, msg):
        try:
            ob = self.buffer.lookup_transform("odom", msg.child_frame_id or "base_link", rclpy.time.Time())
        except Exception:
            return
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        q_wb = (o.x, o.y, o.z, o.w)
        t_wb = (p.x, p.y, p.z)
        r = ob.transform.rotation
        t = ob.transform.translation
        q_ob = (r.x, r.y, r.z, r.w)
        q_bo = q_inv(q_ob)
        t_bo = tuple(-c for c in q_rot(q_bo, (t.x, t.y, t.z)))
        q_wo = q_mul(q_wb, q_bo)
        rt = q_rot(q_wb, t_bo)
        t_wo = (t_wb[0] + rt[0], t_wb[1] + rt[1], t_wb[2] + rt[2])

        out = TransformStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "world"
        out.child_frame_id = "odom"
        out.transform.translation.x, out.transform.translation.y, out.transform.translation.z = t_wo
        (out.transform.rotation.x, out.transform.rotation.y,
         out.transform.rotation.z, out.transform.rotation.w) = q_wo
        self.broadcaster.sendTransform(out)


def main():
    rclpy.init()
    node = WorldTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
