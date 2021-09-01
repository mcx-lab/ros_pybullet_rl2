#!/bin/bash

# installing pip dependencies and make workspace

set -e
echo "Installing dependencies for $ROS_DISTRO"

#  pip stuffs
pip install --upgrade pip
cd /root/pybullet-gym && pip install -e . 
pip install pybullet 
pip install opencv-python
pip install -U --ignore-installed wrapt && pip install tensorflow==1.14
pip install stable-baselines[mpi]
pip install pyyaml
pip install optuna
pip install scikit-image
pip install netifaces
pip install rospkg
pip install squaternion
pip install defusedxml
pip install matplotlib
pip install -U scipy 
pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04 \
wxPython
#  rename ros cv2 
cd /opt/ros/kinetic/lib/python2.7/dist-packages && \
mv cv2.so cv2_renamed.so

# make workspace
cd /root/gail_ws
catkin_make
echo -e "\nsource /root/gail_ws/devel/setup.bash" >> /root/.bashrc
. /root/.bashrc
echo "Finished installing! "