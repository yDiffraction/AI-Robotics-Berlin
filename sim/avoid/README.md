# Obstacle avoidance for the low-level Go2 dodge

Unitree's own "don't walk into things" lives inside its sport-mode controller and is gone once our controller drives the motors. This folder adds it back around the SAFE dodge and tests it in MuJoCo against the real Nav2 Collision Monitor.

## What runs where

```
Go2 L1 lidar (/utlidar/cloud, frame utlidar_lidar, 11 Hz)
   │   + IMU quaternion + joint angles, from a short history at the frame's capture time
   ▼
guard.py (ours, numpy only)
   ├─ LidarToFootprint: lidar -> base_footprint (URDF radar_joint, IMU tilt, leg-kinematic height)
   ├─ self_and_floor: drop floor (<5 cm), trunk column incl. head, leg segments (7 cm)
   ├─ RecentFrames(3) -> side_room / choose_side: dodge only to a side with >= 0.80 m of room
   │      whose floor was observed in every 10 cm strip and every front/middle/rear third
   ├─ keep_off: after the dodge, back away (0.3 m/s) from an obstacle < 20 cm on the dodge side
   └─ Watchdog: velocity passes only if the monitor answered this cycle and the lidar is < 0.3 s old
   │
   ├─ /avoid/cloud_obstacles (PointCloud2, base_footprint)  ─┐
   ├─ /avoid/footprint (PolygonStamped, ±0.47 x ±0.30 m)   ─┤
   └─ /cmd_vel_raw (Twist: the dodge velocity)              ─┴─> Nav2 Collision Monitor (unmodified,
                                                                  approach zone, 0.8 s horizon)
                                                                       │
                                          /cmd_vel (reduced velocity) ─┘ -> Watchdog -> WTW policy
```

Nav2 Collision Monitor ([docs](https://docs.nav2.org/configuration/packages/collision_monitor/configuring-collision-monitor-node.html)) is the proven part, used as-is from `ros-humble-nav2-collision-monitor` 1.1.20 with [collision_monitor.yaml](collision_monitor.yaml). Everything else is ours and unit-tested. The hackathon's `obstacle_detection` code was not reused.

## Things the tests found (would have broken this on the robot)

| Finding | Evidence | Handling |
|---|---|---|
| **Humble's "approach" zone with a static `points` footprint silently never triggers** | [cm_probe.py](cm_probe.py): 1.0 m/s passed straight into a wall | footprint published on `/avoid/footprint`; then 1.0 → 0.425 m/s at 0.35 m gap, as expected |
| **Humble ignores stale lidar data instead of stopping** | CM log: "timestamps differ … Ignoring the source", velocity passed | our Watchdog stops on lidar older than 0.3 s |
| **Humble zeroes all motion once a point is inside the footprint, even motion away** | keep-off commands came back as 0 | keep-off goes around the monitor (still behind the lidar watchdog) |
| A "stop" zone stops *publishing* rather than sending zero | cm_probe | Watchdog treats silence as zero |
| Lidar frames must be interpreted with the robot state **at capture time** | with current state: 544 false points in 30 frames (body rocking, legs swinging 10 cm in 50 ms); with capture-time state: 3 in 44 frames | `StateHistory` |
| Floor under the trunk/head pokes above 5 cm (near-vertical rays put range noise into height) | leak points at x 0.39–0.45 m, z 5.3 cm | under-body column excluded up to the head front |
| One frame leaves a floor strip unobserved in 3–6 % of frames, vetoing a free side | single vs 3-frame windows: 94–97 % → 100 % | `RecentFrames(3)` |
| After a dodge that was cut short, the policy creeps ~0.1 m/s towards the obstacle with a zero command; a swinging foot trips the dog | pop-up traces | `keep_off` (only away from the dodge side; an unrestricted version once pushed towards the obstacle) |
| Shortening the 1 s post-dodge trot (to stop the creep) breaks the torque limit | 1/200 and 2/200 harsh runs at 0.5 s / 0.3 s, vs 0/200 at 1.0 s | kept 1.0 s |
| Test-bench pitfalls (not robot issues) | FastDDS localhost discovery fails in WSL; CycloneDDS drops IP-fragmented samples on WSL loopback (0/55 delivered); `mj_geomDistance` returns 0 for box pairs 0.6 m apart; best-effort clouds dropped when the sim runs faster than real time | CycloneDDS + sub-MTU datagrams ([cyclonedds_wsl.xml](cyclonedds_wsl.xml)), clearance from contact margins, real-time pacing around the dodge |

## Results

See [../results/avoid_final.log](../results/avoid_final.log) and `avoid_final.json`. Every run uses harsh randomized physics: sensor noise, joint clamp, random gait phase, +3 kg, weak motors, 30 ms latency, friction 0.35–1.1.

Final suite: 290 runs, seed 90210, horizon 0.8 s, footprint ±0.30 m, keep-off on.

| Scenario family | Runs | Falls | Contacts | Side choice | Closest approach | Notes |
|---|---|---|---|---|---|---|
| free space | 25 | 0 | 0 | 25/25 | – | none slower than without the monitor (paired runs) |
| wall on one side (room 0.05–1.6 m) | 30 | 0 | 0 | 28/30 | 3.0 cm | the 2 misses are "no dodge" (open side's floor not confirmed), never a blocked side |
| walls on both sides | 30 | 0 | 0 | 29/30 | 0.2 cm | 1 conservative "no dodge"; closest is a 9 cm wall with no dodge and trot drift |
| **forced into a wall** (side check off: monitor alone) | 30 | 0 | 0 | – | 9.5 cm | Collision Monitor stops every time |
| lidar dropout mid-dodge, forced | 15 | 0 | 0 | – | 11.4 cm | Watchdog: "lidar stale" → zero |
| monitor not answering | 5 | 0 | 0 | – | – | max sideways motion 3 cm |
| low box (12 cm) | 15 | 0 | 0 | 15/15 | 8.5 cm | |
| thin pole (5 cm) | 20 | 0 | 1 (−4 mm) | 16/20 | – | the 4 misses are poles the lidar did not see |
| box beside the hind legs | 20 | 0 | 0 | 6/20 | 3.7 cm | **blind spot**: 14 wrong choices, all unseen boxes; no contact only thanks to geometry and the monitor |
| **box pops up in the path 0–0.3 s after the trigger, 0.25–0.8 m away** (adversarial) | 40 | **3** | 7 | – | – | sensing delay + braking distance; see limits |
| same, keep-off disabled (comparison) | 20 | 1 | 3 | – | – | |

**Static or pre-existing obstacles that the lidar can see: 0 falls and 0 contacts in 145 runs.** The software never chose a side that it had measured as blocked. Every wrong choice was an obstacle the sensor never returned. Removing remaining risk needs more sensing, not more tuning.

### Known limits (not fixable in software with this sensor)
- **Blind spot beside and behind the hind legs.** The L1's view tilts 15° down towards the rear. From its 0.25 m mount height, a box 0.8 m behind-beside the dog is hit only about 4 cm above the floor, and those returns count as floor. The side check misses such objects (`rear_box`). The suite saw no contacts, but only because of the geometry. **Operating rule: keep the area beside the hind legs clear, or add a sensor.**
- **Obstacles that appear in the path during the first ~0.3 s of the dodge, within ~0.5 m.** These can be reached anyway. Detection takes 0.1–0.25 s (11 Hz frames plus latency, little side coverage), and braking from ~0.85 m/s takes ~0.2 m. In the pop-up family this caused contacts and some falls. This is physics plus the sensor, not a tuning issue: horizons of 0.8, 1.2 and 1.5 s and footprints of ±0.25–0.35 m all gave the same failing runs.
- **Thin objects (5 cm poles)** return few points and can be missed by the side check.
- **Drift while trotting in place before and after the dodge** is not something the monitor can stop (the command is already zero).

## Run

```bash
# WSL Ubuntu-22.04, once:
apt-get install -y ros-humble-nav2-collision-monitor ros-humble-nav2-lifecycle-manager ros-humble-rmw-cyclonedds-cpp
python3 -m venv --system-site-packages /root/go2_sim/venv_ros && . /root/go2_sim/venv_ros/bin/activate
pip install "numpy<2" mujoco==3.3.7 && pip install torch --index-url https://download.pytorch.org/whl/cpu

# each time (private ROS domain, localhost only; avoid_sim sets CycloneDDS + its WSL config itself):
source /opt/ros/humble/setup.bash && . /root/go2_sim/venv_ros/bin/activate
cd /mnt/c/<path-to-repo>/sim/avoid
python test_guard.py                         # guard unit tests vs MuJoCo ground truth (also runs on Windows)
python cm_probe.py collision_monitor.yaml 0.6 1.0     # monitor alone: wall at 0.6 m, 1 m/s request
python test_avoid.py --tag final             # full scenario suite (~15 min)
python test_avoid.py --families popup,forced --n 20 --params variants/cm_tbc1.2.yaml --footprint-y 0.35
```

## On the real robot
- **Lidar:** `/utlidar/cloud` (frame `utlidar_lidar`). Check that it keeps publishing after sport mode is released.
- **Robot state:** IMU quaternion and joint angles from `rt/lowstate`, kept in `StateHistory` with timestamps.
- **Monitor:** run Collision Monitor with this yaml and publish `/avoid/footprint`. Activate it with the lifecycle manager or the `change_state` calls in `avoid_sim.Bench._activate`.
- **CycloneDDS:** use the Go2's own Cyclone config, not `cyclonedds_wsl.xml`.
- **Kria:** the Yocto image has no Nav2. It needs cross-compiling (`nav2_collision_monitor` with `nav2_util`, `nav2_costmap_2d`, …), or run the monitor on the laptop.
