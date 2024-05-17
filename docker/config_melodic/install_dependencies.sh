#!/bin/bash

# installing pip dependencies and make workspace

set -e
echo "Installing dependencies for $ROS_DISTRO"
echo "Make sure you are in a virtualenv"

# #  pip stuffs
pip install --upgrade pip
cd /root/pybullet-gym && pip install -e . 
pip install stable-baselines3[extra]
pip install -r /root/rl_ws/src/ros_pybullet_rl2/requirements.txt

#  rename ros cv2 
# cd /opt/ros/melodic/lib/python2.7/dist-packages && \
# mv cv2.so cv2_renamed.so ### don't need as long as cv2.so not found
sudo cp /root/rl_ws/src/ros_pybullet_rl2/common/other_packages/roslaunch/loader.py /opt/ros/melodic/lib/python2.7/dist-packages/roslaunch/loader.py

# make workspace
cd /root/rl_ws
catkin_make
echo -e "\nsource /root/rl_ws/devel/setup.bash" >> /root/.bashrc
# . /root/.bashrc
echo "Finished installing! "
echo "Run source .bashrc"