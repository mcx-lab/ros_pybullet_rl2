#!/usr/bin/env python3

"""
Laser scanner simulation based on pybullet rayTestBatch function
This code does not add noise to the laser scanner readings
"""
import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan

class laserScannerModular:
    def __init__(self, pybullet, robot, **kargs):
        # get "import pybullet as pb" and store in self.pb
        self.pb = pybullet
        # get robot from parent class
        self.robot = robot
        # laser params
        self.number_of_lidars = rospy.get_param('~number_of_lidars', 0)
        self.lidar_topics = rospy.get_param('~lidar_topics', None) # scan1, scan2
        if (self.number_of_lidars != len(self.lidar_topics)):
          rospy.logwarn('Number of lidars do not match number of lidar topics, please check input parameters in yaml file.')
        self.laser_param_dic = {}
        ''' laser_param_dic = {scan1: {laser_frame_id: str, 
                                        pb_laser_link_id: int,
                                        angle_min: float,
                                        angle_max: float,
                                        num_Rays: float,
                                        beam_visualisation: bool,
                                        pub_laser_scanner = rospy.Publisher
                                        laser_msg: LaserScan(), # assign range_min, range_max, angle_min, angle_max, header_stamp (last),
                                                                #        angle_increment, header_frame_id, time_increment, scan_time
                                        rayFrom: [float],
                                        rayTo: [float],
                                        count: int,
                                        _____________________________________
                                        ranges: [float],
                                        }
        
        
        }
        '''
        self.publisher_list = []
        # self.count = 0

        for topic in self.lidar_topics:
            self.laser_param_dic[topic] = {}
            self.laser_param_dic[topic]['laser_frame_id'] = rospy.get_param('~laser_%s/frame_id' %topic[-1], None) # laser reference frame, has to be an existing link
            if not self.laser_param_dic[topic]['laser_frame_id']:
                rospy.logerr('required parameter laser_frame_id not set, will exit now')
                rospy.signal_shutdown('required param laser_frame_id not set')
                return
            # get pybullet laser link id from its name
            link_names_to_ids_dic = kargs['link_ids']
            if not self.laser_param_dic[topic]['laser_frame_id'] in link_names_to_ids_dic:
                rospy.logerr('laser reference frame "{}" not found in URDF model, cannot continue'.format(self.laser_param_dic[topic]['laser_frame_id']))
                rospy.logwarn('Available frames are: {}'.format(link_names_to_ids_dic))
                rospy.signal_shutdown('required param frame id not set properly')
                return
            self.laser_param_dic[topic]['pb_laser_link_id'] = link_names_to_ids_dic[self.laser_param_dic[topic]['laser_frame_id']]

            # create laser msg placeholder for publication
            self.laser_param_dic[topic]['laser_msg'] = LaserScan()
            # laser field of view
            angle_min = rospy.get_param('~laser_%s/angle_min' %topic[-1], -1.5707963) 
            angle_max = rospy.get_param('~laser_%s/angle_max' %topic[-1], 1.5707963)
            assert(angle_max > angle_min)
            self.laser_param_dic[topic]['numRays'] = rospy.get_param('~laser_%s/num_beams' %topic[-1], 50) # should be 512 beams but simulation becomes slow
            self.laser_param_dic[topic]['laser_msg'].range_min = rospy.get_param('~laser_%s/range_min' %topic[-1], 0.01)
            self.laser_param_dic[topic]['laser_msg'].range_max = rospy.get_param('~laser_%s/range_max' %topic[-1], 50.0)
            self.laser_param_dic[topic]['beam_visualisation'] = rospy.get_param('~laser_%s/beam_visualisation' %topic[-1], False)
            self.laser_param_dic[topic]['laser_msg'].angle_min = angle_min
            self.laser_param_dic[topic]['laser_msg'].angle_max = angle_max
            self.laser_param_dic[topic]['laser_msg'].angle_increment = (angle_max - angle_min) / self.laser_param_dic[topic]['numRays']
            # register this node in the network as a publisher in /scan topic
            self.laser_param_dic[topic]['pub_laser_scanner'] = rospy.Publisher('scan%s' %topic[-1], LaserScan, queue_size=1)
            self.laser_param_dic[topic]['laser_msg'].header.frame_id = self.laser_param_dic[topic]['laser_frame_id']
            self.laser_param_dic[topic]['laser_msg'].time_increment = 0.01 # ?
            self.laser_param_dic[topic]['laser_msg'].scan_time = 0.1 # 10 hz
            # fixed_joint_index_name_dic = kargs['fixed_joints']
            # compute rays end beam position
            self.laser_param_dic[topic]['rayFrom'], self.laser_param_dic[topic]['rayTo'] = self.prepare_rays(topic)
            # variable used to run this plugin at a lower frequency, HACK
            self.laser_param_dic[topic]['count'] = 0

        self.rayHitColor = [1, 0, 0] # red color
        self.rayMissColor = [0, 1, 0] # green color

    def prepare_rays(self, topic):
        """assume laser is in the origin and compute its x, y beam end position"""
        # prepare raycast origin and end values
        rayFrom = []
        rayTo = []
        for n in range(0, self.laser_param_dic[topic]['numRays']):
            alpha = self.laser_param_dic[topic]['laser_msg'].angle_min + n * self.laser_param_dic[topic]['laser_msg'].angle_increment
            rayFrom.append([self.laser_param_dic[topic]['laser_msg'].range_min * math.cos(alpha),
                          self.laser_param_dic[topic]['laser_msg'].range_min * math.sin(alpha), 0.0])
            rayTo.append([self.laser_param_dic[topic]['laser_msg'].range_max * math.cos(alpha),
                          self.laser_param_dic[topic]['laser_msg'].range_max * math.sin(alpha), 0.0])
        return rayFrom, rayTo

    def transform_rays(self, laser_position, laser_orientation, topic):
        """transform rays from reference frame using pybullet functions (not tf)"""
        laser_position = [laser_position[0], laser_position[1], laser_position[2]]
        TFrayFrom = []
        TFrayTo = []
        rm = self.pb.getMatrixFromQuaternion(laser_orientation)
        rotation_matrix = [[rm[0], rm[1], rm[2]],[rm[3], rm[4], rm[5]],[rm[6], rm[7], rm[8]]]
        for ray in self.laser_param_dic[topic]['rayFrom']:
            position = np.dot(rotation_matrix, [ray[0], ray[1], ray[2]]) + laser_position
            TFrayFrom.append([position[0], position[1], position[2]])
        for ray in self.laser_param_dic[topic]['rayTo']:
            position = np.dot(rotation_matrix, [ray[0], ray[1], ray[2]]) + laser_position
            TFrayTo.append([position[0], position[1], position[2]])
        return TFrayFrom, TFrayTo

    def execute(self):
        """this function gets called from pybullet ros main update loop"""
        # run at lower frequency, laser computations are expensive
        for topic in self.lidar_topics:
            self.laser_param_dic[topic]['count'] += 1
            if self.laser_param_dic[topic]['count'] < 5: # needs this to control publishing frequency of /scan message. 
                pass
            else: 
                self.laser_param_dic[topic]['count'] = 0 # reset count
                # remove any previous laser data if any
                self.laser_param_dic[topic]['laser_msg'].ranges = []
                # remove previous beam lines from screen
                if self.laser_param_dic[topic]['beam_visualisation']:
                    self.pb.removeAllUserDebugItems()
                # get laser link position
                laser_state = self.pb.getLinkState(self.robot, self.laser_param_dic[topic]['pb_laser_link_id'])
                # transform start and end position of the rays which were generated considering laser at the origin
                rayFrom, rayTo  = self.transform_rays(laser_state[0], laser_state[1], topic) # position + orientation
                # raycast using 4 threads
                results = self.pb.rayTestBatch(rayFrom, rayTo, 4)
                for i in range(self.laser_param_dic[topic]['numRays']):
                    if self.laser_param_dic[topic]['beam_visualisation']:
                        hitObjectUid = results[i][0]
                        if hitObjectUid < 0:
                            # draw a line on pybullet gui for debug purposes in green because it did not hit any obstacle
                            self.pb.addUserDebugLine(rayFrom[i], rayTo[i], self.rayMissColor)
                        else:
                            # draw a line on pybullet gui for debug purposes in red because it hited obstacle, results[i][3] -> hitPosition
                            self.pb.addUserDebugLine(rayFrom[i], results[i][3], self.rayHitColor)
                    # compute laser ranges from hitFraction -> results[i][2]
                    self.laser_param_dic[topic]['laser_msg'].ranges.append(results[i][2] * self.laser_param_dic[topic]['laser_msg'].range_max)
                # update laser time stamp with current time
                self.laser_param_dic[topic]['laser_msg'].header.stamp = rospy.Time.now()
                # publish scan
                self.laser_param_dic[topic]['pub_laser_scanner'].publish(self.laser_param_dic[topic]['laser_msg'])

