#!/usr/bin/python3

"""
NOTE: this node can be safely deleted if tf is broadcasted from pybullet_ros.py itself
"""
################################ IF I MAKE USE OF IRA_LASER_TOOL, I do not need this node #######################
import rospy
import tf

from sensor_msgs.msg import LaserScan

class lidarToTF(object):
    '''
    subscribe to odom topic and publish tf
    '''
    def __init__(self):
        self.br = tf.TransformBroadcaster()
        self.number_of_lidars = rospy.get_param('~number_of_lidars', 0)
        self.lidar_topics = rospy.get_param('~lidar_topics', None)
        if (number_of_lidars != len(lidar_topics)):
          rospy.logwarn('Number of lidars do not match number of lidar topics, please check input parameters in yaml file.')
        self.laser_param_dic = {}
        self.subscriber_list = []
        self.count = 0

        for topic in lidar_topics:
          self.count += 1
          self.subscriber_list.append(rospy.Subscriber(topic, LaserScan, self.lidarCallback, queue_size=1))
          self.laser_param_dic[topic] = []
          # laser_param_dic[topic] is a list of [trans_x, trans_y, trans_z, orien_x, orien_y, orien_z, orien_w]
          self.laser_param_dic[topic].append(rospy.get_param('~laser_%s/trans_x_tf_to_base_link' %str(count), 0))
          self.laser_param_dic[topic].append(rospy.get_param('~laser_%s/trans_y_tf_to_base_link' %str(count), 0))
          self.laser_param_dic[topic].append(rospy.get_param('~laser_%s/trans_z_tf_to_base_link' %str(count), 0))
          self.laser_param_dic[topic].append(rospy.get_param('~laser_%s/rot_x_tf_to_base_link' %str(count), 0))
          self.laser_param_dic[topic].append(rospy.get_param('~laser_%s/rot_y_tf_to_base_link' %str(count), 0))
          self.laser_param_dic[topic].append(rospy.get_param('~laser_%s/rot_z_tf_to_base_link' %str(count), 0))
          self.laser_param_dic[topic].append(rospy.get_param('~laser_%s/rot_w_tf_to_base_link' %str(count), 0))

    def lidarCallback(self, laser_msg):
        laser_id = laser_msg.header.frame_id # i.e. laser_1, laser_2
        translation = [self.laser_param_dic['scan%s'][0] %laser_id[-1], self.laser_param_dic['scan%s'][1] %laser_id[-1], self.laser_param_dic['scan%s'][2] %laser_id[-1]]
        rotation = [self.laser_param_dic['scan%s'][3] %laser_id[-1], self.laser_param_dic['scan%s'][4] %laser_id[-1], self.laser_param_dic['scan%s'][5] %laser_id[-1]]
        # translation, rotation, time, child, parent
        self.br.sendTransform(translation, rotation, rospy.Time.now(), laser_id, 'base_link')

if __name__ == '__main__':
    rospy.init_node('lidar_tf_broadcaster', anonymous=False)
    lidar_tf_broadcaster = lidarToTF()
    rospy.spin()
