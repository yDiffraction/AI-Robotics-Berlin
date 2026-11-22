# Go2 dodge (ROS 2, trigger only)

Minimal dodge for the real Go2: one node that moves the robot out of the way when you press **X on the Go2 remote**
(or when `/reflex/dash` arrives), then walks it slowly back to where it stood, plus a keyboard node that sends
`/reflex/dash`. No perception, no build step: plain `rclpy` scripts.

Default dodge: **back at 1.1 m/s for 0.5 s**, stand 1 s, then **back to the start at 0.25 m/s**. The way back uses the
robot's odometry (`/sportmodestate`) and stops within 5 cm of the start; without odometry it walks back 80 % of the
commanded distance by time, so it never overshoots towards the thrower.

Needs ROS 2 Humble with the `unitree_api` and `unitree_go` messages (in the go2-kria overlay,
`starter/go2-kria/src/unitree_api`, `.../unitree_go`) and the Go2 on the same DDS network (domain 0, CycloneDDS;
on the Kria: `ROS_DOMAIN_ID=0`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
`CYCLONEDDS_URI=file:///etc/cyclonedds/cyclonedds-go2.xml`).

```bash
source /opt/ros/humble/setup.bash && source <go2-kria overlay>/local_setup.bash
python3 dodge/go2_dodge.py --ros-args -p dry_run:=true      # first: log only, press X and watch the log
python3 dodge/go2_dodge.py                                  # real: X on the remote = dodge
python3 dodge/dodge_keys.py                                 # optional, second terminal: d = dodge, space = STOP, q = quit
```

On the Kria of the Go2 (it reaches the Go2 on 192.168.123.99): install the starter overlay once
(`starter/go2-kria/kria-runtime.tar.gz`, extracted to `/opt/go2-dodge`), copy `go2_dodge.py`, `dodge_keys.py` and
`kria/*.sh` to `/root/dodge/`, then from the laptop:

```bash
ssh root@192.168.77.1 /root/dodge/start.sh                    # runs in the background, log /root/dodge/dodge.log
ssh root@192.168.77.1 /root/dodge/start.sh -p dry_run:=true   # same, log only
ssh root@192.168.77.1 tail -f /root/dodge/dodge.log
ssh root@192.168.77.1 /root/dodge/stop.sh
```

The X button is read from `/lowstate` (`wireless_remote` bytes 2-3) and from `/wirelesscontroller`: on our Go2 only
`/lowstate` carries the remote, `/wirelesscontroller` stays silent.

Trigger and watch from any shell or agent on the same ROS network:

```bash
ros2 topic pub --once /reflex/dash std_msgs/msg/Empty "{}"   # dodge
ros2 topic pub --once /dodge/stop std_msgs/msg/Empty "{}"    # stop at once
ros2 topic echo /dodge/event                                 # dash / hold / return / idle with measured distances
ros2 topic echo /wirelesscontroller --field keys             # X pressed = bit 10 (1024) set
```

Parameters (`--ros-args -p name:=value`): `button` (x; a, b, y, l1, r1, ... or none), `direction`
back|left|right|forward (back), `speed` m/s (1.1, max 1.5), `duration` s (0.5, max 2), `come_back` (true), `hold` s
(1.0), `return_speed` m/s (0.25, max 0.5), `tolerance` m (0.05), `return_scale` (0.8, without odometry), `refractory`
s (1.0), `odom` sportmodestate|odometry|none, `odom_topic`, `controller_topic` (/wirelesscontroller), `dry_run`
(false). A trigger during a running dodge is ignored; every phase has a time limit; Ctrl+C sends StopMove.
Keep the space behind the robot free: there is no obstacle check. The remote's own button functions still work
as usual; X alone does nothing in the default sport mode as far as we know, check once with `dry_run`.

Gazebo sim (test container, `odom:=odometry -p odom_topic:=/odom/ground_truth_fast`, X pressed via a fake
`/wirelesscontroller` message): commanded 1.1 m/s for 0.5 s gives 0.27 m back in the CHAMP model (slow
acceleration there), the way back by odometry ended 2 cm from the start, max tilt 9 deg.

## Robot launch with SNN and web UI (`robot.launch.py`)

One command starts both nodes: `go2_dodge.py` (X = manual dodge) and `snn_bridge.py` (event camera -> SNN on the
FPGA -> throw detection -> `/reflex/dash` only while **auto-dodge** is on; **Y** on the remote, the switch on the
web page or `/snn/auto_set` toggle it; it is off at start). Web UI on port 8080: event camera, SNN spikes, readout
with threshold (slider), THROW DETECTED banner, robot state, status pills, optional Go2 color camera (checkbox).

```bash
source /root/dodge/env.sh && ros2 launch /root/dodge/robot.launch.py              # on the Kria, open http://192.168.77.1:8080
ros2 launch dodge/robot.launch.py dry_run:=true                                   # everything live, robot only logs
ros2 launch dodge/robot.launch.py camera:=replay:/abs/rec.bin snn:=mock dry_run:=true   # desk test, no board
python3 dodge/snn_bridge.py --no-ros --ros-args -p camera:=replay:rec.bin -p snn:=mock  # UI test without ROS
```

`ros2 launch dodge/robot.launch.py --show-args` lists all arguments (camera local|ssh|listen|replay:<file>, snn
fpga|mock, spleat_host, theta, auto, web_port, color, dry_run, speed, return_speed, odom, ...). The readout of the
current FPGA model (Tim's v3 early throw-motion detector, channel 0, y = 0.85 y + 0.15 spike, theta 0.85,
provisional) is in `model/readout.json`. `web/?demo=1` shows the UI with synthetic data, no backend needed.
Tests: `python3 dodge/tests/test_snn_bridge.py`.
