# Native (arm64 on Apple Silicon, amd64 elsewhere) ROS 2 Humble for desk simulation.
# The Kria cross-build image (durance/go2-yocto) is separate and unrelated.

# --- sil: build and run the Durance ROS 2 pipeline against simulated hardware ---
FROM ros:humble-ros-base-jammy AS sil
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake nlohmann-json3-dev \
        python3-colcon-common-extensions \
        ros-humble-rmw-cyclonedds-cpp ros-humble-cyclonedds \
        ros-humble-rosidl-generator-dds-idl \
    && rm -rf /var/lib/apt/lists/*
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_DOMAIN_ID=0 \
    WS=/ws
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
WORKDIR /ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# --- gazebo: adds Gazebo Classic, the CHAMP-based Go2 model and a browser desktop ---
FROM sil AS gazebo
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control \
        ros-humble-ros2-control ros-humble-ros2-controllers \
        ros-humble-xacro ros-humble-robot-localization \
        ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
        ros-humble-velodyne-description ros-humble-velodyne-gazebo-plugins \
        ros-humble-rviz2 ros-humble-teleop-twist-keyboard \
        tigervnc-standalone-server tigervnc-tools novnc websockify fluxbox xterm \
        mesa-utils libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
# Third-party Go2 simulation (CHAMP controller + Gazebo world), built into /opt/go2_sim
RUN mkdir -p /opt/go2_sim/src && cd /opt/go2_sim/src \
    && git clone --depth 1 --branch humble https://github.com/anujjain-dev/unitree-go2-ros2.git \
    && cd /opt/go2_sim && apt-get update \
    && (rosdep update --rosdistro humble || true) \
    && rosdep install --from-paths src --ignore-src -r -y --rosdistro humble \
    && rm -rf /var/lib/apt/lists/* \
    && . /opt/ros/humble/setup.sh \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log
ENV GO2_SIM=/opt/go2_sim \
    LIBGL_ALWAYS_SOFTWARE=1 \
    DISPLAY=:1
COPY desktop.sh /usr/local/bin/desktop.sh
RUN chmod 755 /usr/local/bin/desktop.sh
