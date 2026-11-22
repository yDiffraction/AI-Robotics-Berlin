# Go2 dodge (ROS 2, trigger only)

Minimal dodge for the real Go2: one node that moves the robot out of the way when `/reflex/dash` arrives,
and a keyboard node to send that trigger. No perception, no build step: plain `rclpy` scripts.

Needs ROS 2 Humble with the `unitree_api` messages (in the go2-kria overlay, `starter/go2-kria/src/unitree_api`)
and the Go2 on the same DDS network (domain 0, CycloneDDS; on the Kria: `ROS_DOMAIN_ID=0`,
`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, `CYCLONEDDS_URI=file:///etc/cyclonedds/cyclonedds-go2.xml`).

```bash
source /opt/ros/humble/setup.bash && source <go2-kria overlay>/local_setup.bash
python3 dodge/go2_dodge.py --ros-args -p dry_run:=true      # first: log only, check the trigger arrives
python3 dodge/go2_dodge.py                                  # real: back off 0.4 m (0.8 m/s, 0.5 s), then walk back
python3 dodge/dodge_keys.py                                 # second terminal: d = dodge, space = STOP, q = quit
```

Trigger from any shell or agent on the same ROS network:

```bash
ros2 topic pub --once /reflex/dash std_msgs/msg/Empty "{}"   # dodge
ros2 topic pub --once /dodge/stop std_msgs/msg/Empty "{}"    # stop at once
ros2 topic echo /dodge/event                                 # what the node did (JSON)
```

Parameters (`--ros-args -p name:=value`): `direction` back|left|right|forward (back), `speed` m/s (0.8, max 1.5),
`duration` s (0.5, max 2), `come_back` (true), `hold` s before coming back (1.0), `return_speed` (0.3),
`refractory` s (1.0), `dry_run` (false). The move is time based (Move 1008 re-sent every 0.1 s, then
StopMove 1003); a dash during a running dodge is ignored. Keep the space around the robot free: there is
no obstacle check. The SNN trigger (or `sim/.../ball_trigger_udp.py`) publishes the same `/reflex/dash`.
