# =============================================================================
# Kinova Gen3 + MoveIt2 + MoveIt Servo — ROS2 Jazzy + Gazebo
# =============================================================================

FROM moveit/moveit2:jazzy-release

ENV DEBIAN_FRONTEND=noninteractive \
      ROS_DISTRO=jazzy \
      WORKSPACE=/ros2_ws

# ── Install build tools ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      python3-vcstool \
      python3-rosdep \
      git \
      build-essential \
      cmake \
      && rm -rf /var/lib/apt/lists/*

# ── Update rosdep ───────────────────────────────────────────────────────────
RUN apt-get update && rosdep update && rm -rf /var/lib/apt/lists/*

# ── Clone ros2_kortex ───────────────────────────────────────────────────────
RUN git clone --depth=1 \
      https://github.com/Kinovarobotics/ros2_kortex.git ${WORKSPACE}/src/ros2_kortex

# ── Import kortex .repos files only ─────────────────────────────────────────
RUN cd ${WORKSPACE}/src \
      && vcs import --skip-existing --input ros2_kortex/ros2_kortex.jazzy.repos \
      && vcs import --skip-existing --input ros2_kortex/ros2_kortex-not-released.jazzy.repos \
      && vcs import --skip-existing --input ros2_kortex/simulation.jazzy.repos

# ── Install dependencies with rosdep ────────────────────────────────────────
RUN apt-get update \
      && rosdep install --ignore-src --from-paths ${WORKSPACE}/src -y -r \
      && rm -rf /var/lib/apt/lists/*

# ── Build kortex only ───────────────────────────────────────────────────────
RUN bash -c "source /opt/ros/jazzy/setup.bash \
      && cd ${WORKSPACE} \
      && colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --parallel-workers 1"

# ── Entrypoint ──────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]