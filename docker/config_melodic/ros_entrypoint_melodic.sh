#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/melodic/setup.bash"
# source "/opt/ros/$ROS_DISTRO/setup.bash"
exec "$@"