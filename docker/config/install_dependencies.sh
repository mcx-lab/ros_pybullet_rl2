#!/bin/bash

# installing pip dependencies and make workspace

set -e
echo "Installing dependencies for $ROS_DISTRO"

# #  pip stuffs
pip install --upgrade pip
cd /root/pybullet-gym && pip install -e . 
# pip install pybullet 
# pip install opencv-python
# pip install -U --ignore-installed wrapt && pip install tensorflow==1.14
# pip install stable-baselines[mpi]
# pip install pyyaml
# pip install optuna
# pip install scikit-image
# pip install netifaces
# pip install rospkg
# pip install squaternion
# pip install defusedxml
# pip install matplotlib
# pip install -U scipy 
# pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04 \
# wxPython
pip install stable-baselines3[extra]
# python -m pip install seaborn
# pip install sb3-contrib
#got error, cant find this file
pip install -r /root/rl_ws/src/ros_pybullet_rl2/requirements.txt

#  rename ros cv2 
cd /opt/ros/kinetic/lib/python2.7/dist-packages && \
mv cv2.so cv2_renamed.so

# make workspace
cd /root/rl_ws
catkin_make
echo -e "\nsource /root/rl_ws/devel/setup.bash" >> /root/.bashrc
# . /root/.bashrc
echo "Finished installing! "
echo "Run source .bashrc"