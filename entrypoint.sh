#!/bin/bash
# Sources the workspace once on container start, then hands off to
# whatever `command:` was set in docker-compose.
set -e
source /ros2_ws/install/setup.bash
exec "$@"