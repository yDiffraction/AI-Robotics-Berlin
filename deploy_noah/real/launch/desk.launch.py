"""Desk test of real_robot.launch.py: the exact same nodes, but the hardware replaced by stand-ins.

  Kria       -> Noah's spleat_emulator (sim/tools), 3 scores per frame like his model on the Kria,
                looping idle 3 s, throw 1 s, idle 3 s, feint 1 s
  Go2 LiDAR  -> Noah's fake_lidar (flat floor; obstacle: ros2 param set /fake_lidar obstacle_distance 0.45)
  Go2 sport  -> the starter's robot_sim (acks every /api/sport/request, no physics)

Run with real/env.sh desk (loopback, domain 73), so nothing reaches the dog even if it is wired:
  run_noah.ps1 desk
"""

import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
EMULATOR = os.path.join(HERE, "..", "..", "sim", "tools", "spleat_emulator.py")
PORT = "3334"


def generate_launch_description():
    robot_sim = os.path.join(get_package_prefix("robot_sim"), "lib", "robot_sim", "robot_sim")
    return LaunchDescription([
        ExecuteProcess(cmd=["python3", "-u", EMULATOR, "--mode", "sequence", "--loop", "--classes", "3",
                            "--idle-class", "2", "--host", "127.0.0.1", "--port", PORT,
                            "--sequence", "2:3,0:1,2:3,1:1"],
                       output="screen", name="spleat_emulator"),
        ExecuteProcess(cmd=[robot_sim], output="screen", name="robot_sim"),
        Node(package="go2_sim_tools", executable="fake_lidar", output="screen"),
        TimerAction(period=1.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(HERE, "real_robot.launch.py")),
            launch_arguments={"kria_host": "127.0.0.1", "port": PORT}.items())]),
    ])
