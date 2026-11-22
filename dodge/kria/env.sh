export COLCON_PYTHON_EXECUTABLE=/usr/bin/python3
source /opt/ros/humble/setup.bash
source /opt/go2-dodge/local_setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file:///etc/cyclonedds/cyclonedds-go2.xml
