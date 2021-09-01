#!/usr/bin/env python3

"""
Query robot base pose and speed from pybullet and publish to /odom topic
This component does not add any noise to it
"""

import rospy
from nav_msgs.msg import Odometry

class simpleOdometry:
    def __init__(self, pybullet, robot, **kargs):
        self.pb = pybullet
        # get robot from parent class
        self.robot = robot
        # register this node as a /odom publisher
        self.pub_odometry = rospy.Publisher('odom', Odometry, queue_size=1)
        # save some overhead by setting some information only once
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = rospy.get_param('~odom_frame', 'odom')
        self.odom_msg.child_frame_id = rospy.get_param('~robot_base_frame', 'base_link')

    def execute(self):
        """this function gets called from pybullet ros main update loop"""
        # set msg timestamp based on current time
        self.odom_msg.header.stamp = rospy.Time.now()
        # query base pose from pybullet and store in odom msg
        [self.odom_msg.pose.pose.position.x,\
         self.odom_msg.pose.pose.position.y,\
         self.odom_msg.pose.pose.position.z],\
        [self.odom_msg.pose.pose.orientation.x,\
         self.odom_msg.pose.pose.orientation.y,\
         self.odom_msg.pose.pose.orientation.z,\
        self.odom_msg.pose.pose.orientation.w] = self.pb.getBasePositionAndOrientation(self.robot)
        # query base velocity from pybullet and store it in msg
        [self.odom_msg.twist.twist.linear.x,\
         self.odom_msg.twist.twist.linear.y,\
         self.odom_msg.twist.twist.linear.z],\
        [self.odom_msg.twist.twist.angular.x,\
         self.odom_msg.twist.twist.angular.y,\
         self.odom_msg.twist.twist.angular.z] = self.pb.getBaseVelocity(self.robot)
        self.pub_odometry.publish(self.odom_msg)
        # tf publication (odom to base_link)
        # TODO: tf can be broadcasted from here, but we have issues in melodic due to python 2/3 ...
        # self.br = tf.TransformBroadcaster() # need to be done from constructor
        # translation, rotation, time, child, parent
        # self.br.sendTransform(position, orientation, rospy.Time.now(), 'base_link', 'odom')
