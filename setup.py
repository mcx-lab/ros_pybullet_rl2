#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# for your packages to be recognized by python
d = generate_distutils_setup(
 packages=['ros_pybullet_rl2'],
 package_dir={'ros_pybullet_rl2': 'src/ros_pybullet_rl2'}
)

#utils = generate_distutils_setup(
# packages=['utils'],
# package_dir={'': 'src/ros_pybullet_rl/utils'}
#)

setup(**d)
#setup(**utils)