#!/usr/bin/env python3

"""
query force torque sensor on robot to /force_torque_sensor
"""

import rospy
from ros_pybullet_rl2.msg import ForceTorqueData
from ros_pybullet_rl2.msg import ForceTorqueSensors


class forceTorqueSensor:
    def __init__(self, pybullet, robot, **kargs):
        self.pb = pybullet
        # get robot from parent class
        self.robot = robot
        # get joints names and store them in dictionary
        self.joint_index_name_dic = kargs['fixed_joints']
        self.ft_sensor_joint_index_name_dic = {}

        sensor_frame_id = rospy.get_param('~force_torque_sensor/sensor_id', None) # sensor reference frame, has to be an existing joint
        if not sensor_frame_id:
            rospy.logerr('required parameter sensor_frame_id not set, will exit now')
            rospy.signal_shutdown('required param sensor_frame_id not set')
            return
        for joint_index in self.joint_index_name_dic:
            if self.joint_index_name_dic[joint_index] in sensor_frame_id:
                self.ft_sensor_joint_index_name_dic[joint_index] = self.joint_index_name_dic[joint_index]
        if len(sensor_frame_id) != len(self.ft_sensor_joint_index_name_dic):
            rospy.logerr('sensor reference frame "{}" not found in URDF model, cannot continue'.format(sensor_frame_id))
            rospy.logwarn('Available frames are: {}'.format(link_names_to_ids_dic))
            rospy.signal_shutdown('required param frame id not set properly')
            return
        for joint_index in self.ft_sensor_joint_index_name_dic:
            self.pb.enableJointForceTorqueSensor(self.robot, joint_index, enableSensor=True)
        self.force_torque_pub = rospy.Publisher('force_torque_sensor', ForceTorqueSensors, queue_size=1)

    def execute(self):
        """this function gets called from pybullet ros main update loop"""
        # setup msg placeholder
        force_torque_sensor = ForceTorqueSensors()
        # get joint states
        for joint_index in self.ft_sensor_joint_index_name_dic:
            # get joint state from pybullet
            joint_state = self.pb.getJointState(self.robot, joint_index)
            # fill msg
            force_torque_msg = ForceTorqueData()
            # force_torque_msg.name.append(self.joint_index_name_dic[joint_index])
            force_torque_msg.name = self.ft_sensor_joint_index_name_dic[joint_index]
            # force_torque_msg.force_torque.append(joint_state[2])
            force_torque_msg.force_torque = joint_state[2]
            force_torque_sensor.sensor.append(force_torque_msg)
            force_torque_msg.header.stamp = rospy.Time.now()

        # publish joint states to ROS
        self.force_torque_pub.publish(force_torque_sensor)
