
Cross compilation with ROS2 and Docker

This documentation assumes that you have already learnt how to setup Kria and SD card from a separate page.
On laptop : cross-compile project

To build on your laptop, run this from the extracted go2-kria directory:

mkdir -p sdk
cp /path/to/sdk_x86_64_installer.sh sdk/installer.sh
./manage.sh image
./manage.sh check
./manage.sh build
./manage.sh package

On Kria

The remaining hardware check is the live Go2 and camera.

./manage.sh package creates kria-runtime.tar.gz. Deploy it by copying that archive to the Kria and extracting it.

    Copy a new build from your laptop

scp kria-runtime.tar.gz root@durance-kria.local:/tmp/
ssh root@durance-kria.local

    Install and test it on the Kria

bash
release=$(mktemp -d /opt/go2-XXXXXX)
tar -xzf /tmp/kria-runtime.tar.gz -C "$release"
echo "$release"

Remember the /opt/go2-... path that echo prints (write it down?)

Then load ROS and run the test:

export COLCON_PYTHON_EXECUTABLE=/usr/bin/python3
source /opt/ros/humble/setup.bash
source "$release/local_setup.bash"
python3 "$release/smoke.py"

The test runs with the robot disconnected. For later SSH sessions, set release to the saved /opt/go2-... path and repeat the export and two source lines. You only need to copy and extract again when deploying a new build.
With robot

    Run with the actual Go2

In the same sourced shell, connect the robot and verify that the Kria receives a live pose:

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file:///etc/cyclonedds/cyclonedds-go2.xml

ip route get 192.168.123.161
ros2 topic echo --once --qos-reliability best_effort /utlidar/robot_pose geometry_msgs/msg/PoseStamped

Once a pose appears, start your usual backward-server and camera client. With the Go2 ready for motion, launch the pipeline using the port your server listens on:

ros2 launch bringup pipeline.launch.py despot:=false port:=41017

