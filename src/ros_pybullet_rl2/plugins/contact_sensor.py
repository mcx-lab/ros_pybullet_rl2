#!/usr/bin/env python3

"""
query contact sensor on robot to /contact_sensor
"""

import rospy
# can use this because same data structure as contact sensors 
from ros_pybullet_rl.msg import ForceTorqueData
from ros_pybullet_rl.msg import ForceTorqueSensors


class contactSensor:
    def __init__(self, pybullet, robot, **kargs):
        self.pb = pybullet
        # get robot from parent class
        self.robot = robot
        # get links names-to-ids and store them in dictionary
        self.contact_sensor_link_name_index_dic = {} # key: link_name, value: link_id or index

        # get pybullet laser link id from its name
        link_names_to_ids_dic = kargs['link_ids']
        self.sensor_frame_id = rospy.get_param('~contact_sensor/sensor_id', None) # sensor reference frame, has to be an existing joint
        for sensor_id in self.sensor_frame_id:
            if not sensor_id in link_names_to_ids_dic:
                rospy.logerr('required parameter contact_sensor/sensor_id not set, will exit now')
                rospy.signal_shutdown('required param contact_sensor/sensor_id not set')
                return
            self.contact_sensor_link_name_index_dic[sensor_id] = link_names_to_ids_dic[sensor_id]

        if len(self.sensor_frame_id) != len(self.contact_sensor_link_name_index_dic):
            rospy.logerr('sensor reference frame "{}" not found in URDF model, cannot continue'.format(sensor_frame_id))
            rospy.logwarn('Available frames are: {}'.format(contact_sensor_link_name_index_dic))
            rospy.signal_shutdown('required param frame id not set properly')
            return
        self.contact_pub = rospy.Publisher('contact_sensor', ForceTorqueSensors, queue_size=1)

    def execute(self):
        """this function gets called from pybullet ros main update loop"""
        # setup msg placeholder
        contact_sensor = ForceTorqueSensors()
        for link_name in self.contact_sensor_link_name_index_dic:
            link_contact_msg = []
            # get link contact points from pybullet, link_contact_state will contain a list of contact points, each contact point contain a list of data. 
            link_contact_state = self.pb.getContactPoints(linkIndexA=self.contact_sensor_link_name_index_dic[link_name])
            # check if there is contact point at all
            for contact_point in range(len(link_contact_state)):
                if link_contact_state[contact_point]:
                    link_contact_msg.append(link_contact_state[contact_point][8])
                    link_contact_msg.append(link_contact_state[contact_point][9])
            ### processing to keep only 1 most high force contact point info
            if len(link_contact_msg) == 0:
                link_contact_msg_final = [0.0,0.0]
            if len(link_contact_msg) > 2:
                separating_d = link_contact_msg[0::2]
                normal_f = link_contact_msg[1::2]
                link_contact_msg_final = [min(separating_d),max(normal_f)] # only contain 1 pair of data
            if len(link_contact_msg) == 2:
                link_contact_msg_final = link_contact_msg
            # fill msg
            contact_msg = ForceTorqueData() # list
            contact_msg.name = link_name
            contact_msg.force_torque = link_contact_msg_final
            contact_msg.header.stamp = rospy.Time.now()
            contact_sensor.sensor.append(contact_msg)

        # publish joint states to ROS
        self.contact_pub.publish(contact_sensor)
