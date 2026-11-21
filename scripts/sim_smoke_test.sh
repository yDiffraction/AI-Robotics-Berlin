#!/usr/bin/env bash
# Headless smoke test for the Go2 CHAMP Gazebo sim (ROS 2 Humble, WSL Ubuntu-22.04)
set +u
source /opt/ros/humble/setup.bash
source /root/go2_ws/install/setup.bash
source /usr/share/gazebo/setup.sh
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}   # keep sim traffic off the robot's domain 0

timeout 90 ros2 launch go2_config gazebo.launch.py gui:=False headless:=True > /root/simtest.log 2>&1 &
LAUNCH=$!
sleep 50

echo "== nodes";        timeout 15 ros2 node list
echo "== topics";       timeout 15 ros2 topic list | grep -E "cmd_vel|odom|joint_states|imu|foot"
echo "== controllers";  timeout 15 ros2 control list_controllers
echo "== joint_states"; timeout 10 ros2 topic echo --once /joint_states --field name
echo "== walk: send cmd_vel 0.3 m/s for 5 s, compare odom"
timeout 5 ros2 topic echo --once /odom/raw --field pose.pose.position 2>/dev/null || timeout 5 ros2 topic echo --once /odom --field pose.pose.position
timeout 5 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}" > /dev/null
timeout 5 ros2 topic echo --once /odom/raw --field pose.pose.position 2>/dev/null || timeout 5 ros2 topic echo --once /odom --field pose.pose.position

kill $LAUNCH 2>/dev/null; wait $LAUNCH 2>/dev/null
pkill -f gzserver; pkill -f ros2-daemon
echo "== errors in launch log"; grep -iE "\[ERROR\]|exception" /root/simtest.log | head -15
echo SMOKE_DONE
