from pybulletgym.envs.yc.robots.py_omni_base import OmniBase
from pybulletgym.envs.yc.robots.py_robot_bases import URDFBasedRobot # like atlas example, with omn_descriptions in apybulletgym/envs/assets/robot
from pybulletgym.envs.yc.robots.py_robot_bases import MJCFBasedRobot # like other examples, with .xml file in apybulletgym/envs/assets/mjcf
import numpy as np
import pybullet as p
from geometry_msgs.msg import Twist, Vector3Stamped, Pose, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from ros_pybullet_rl.msg import DepthImageProcessed
import rospy
import math
import random
from squaternion import Quaternion
from skimage.measure import block_reduce

from time import sleep 

import pybullet_data



class PyOmnirobot(OmniBase, URDFBasedRobot):
    random_yaw = False
    # Node ID: 1 - front left wheels, 2 - back, 3 - front right. 
    foot_list = ["front_left_wheel", "back_wheel","front_right_wheel"] # this is the link name of the locomotor in contact with the ground. To be tracked. 
    # rename foot_list to the 3 wheels link name in contact with the ground of the Omnirobot. 
    def __init__(self, goal_x, goal_y):
        start_pos_x = 0.0
        OmniBase.__init__(self, goal_x=0, goal_y=0, start_pos_x=start_pos_x)
        # consider using rospy.get_param to get inputs for max_vel vehicle can travel at, and feed this input into URDFBasedRobot where the action space is defined. 
        URDFBasedRobot.__init__(self, "omnirobot_v3/urdf/omnirobot_v3_1LiDAR.urdf",
                                "Omnirobot", 
                                action_dim=3, 
                                obs_dim=29 # original 29, 23 is exclude contact_data
                                )

        # Initialise variables
        self.odom_info = np.zeros((13,), dtype=np.float32) # list of 13 floats. 
        self.laser_info = np.full((1083,), 31.0, dtype=np.float32) # was list of 1083 floats. 
        self.depth_info = np.zeros((307200,), dtype=np.float32) # list of 307200 floats.
        self.contact_info = np.zeros((6,), dtype=np.float32) # list of 6 floats. !!! TAKE NOTE THE ORDER OF THIS IS IMPORTANT IN GENERALISING PROGRAM !!! 
        self.cmd_vel_info = np.zeros((3,), dtype=np.float32) # list of 3 floats. 

        # For specific goal change
        self.goal_change = 0
        self.goal_count = 0
        self.goal_repeat = 0
        self.goal_random_disp = 0
        self.select_env = rospy.get_param("~select_env")
        self.is_validate = rospy.get_param("~is_validate")
        if not self.is_validate:
            self.goal_set = eval(rospy.get_param("~env_obj{}/goal_set".format(self.select_env)))
        else: 
            self.goal_set = eval(rospy.get_param("~env_obj{}/validation_goal_set".format(self.select_env)))

        self.action_set = {'vel_x':[-0.2655, 0.2655], 
                        'vel_y':[-0.2655, 0, 0.2655], 
                        'vel_th': [-0.2, 0, 0.2]}
        # self.reward_range = (-np.inf, np.inf)
        self.max_vel_x = 0.51
        self.max_vel_y = 0.46
        self.max_vel_th = 1.00
        self.min_vel_x = -0.51
        self.min_vel_y = -0.46
        self.min_vel_th = -1.00
        self.max_vel_all = 0.2655

    def get_properties(self):
        """
        construct 3 dictionaries:
        - joint index to joint name x2 (1 for revolute, 1 for fixed joints)
        - link name to link index dictionary
        """
        rev_joint_index_name_dic = {}
        fixed_joint_index_name_dic = {}
        link_names_to_ids_dic = {}
        for joint_index in range(0, self._p.getNumJoints(self.objects)):
            info = self._p.getJointInfo(self.objects, joint_index) # self._p.getJointInfo(self.robot_loadedURDF, joint_index)
            # rospy.loginfo(info)
            # build a dictionary of link names to ids
            link_names_to_ids_dic[info[12].decode('utf-8')] = joint_index
            # ensure we are dealing with a revolute joint
            if info[2] == self._p.JOINT_REVOLUTE:
                # insert key, value in dictionary (joint index, joint name)
                rev_joint_index_name_dic[joint_index] = info[1].decode('utf-8') # info[1] refers to joint name
            elif info[2] == self._p.JOINT_FIXED:
                # insert key, value in dictionary (joint index, joint name)
                fixed_joint_index_name_dic[joint_index] = info[1].decode('utf-8') # info[1] refers to joint name
        return rev_joint_index_name_dic, fixed_joint_index_name_dic, link_names_to_ids_dic


    '''def odom_callback(self, odom_data):
        self.odom_info = np.array([odom_data.pose.pose.position.x,odom_data.pose.pose.position.y,\
            odom_data.pose.pose.position.z,odom_data.pose.pose.orientation.x,\
            odom_data.pose.pose.orientation.y,odom_data.pose.pose.orientation.z,\
            odom_data.pose.pose.orientation.w,odom_data.twist.twist.linear.x,\
            odom_data.twist.twist.linear.y,odom_data.twist.twist.linear.z,\
            odom_data.twist.twist.angular.x,odom_data.twist.twist.angular.y,\
            odom_data.twist.twist.angular.z], dtype=np.float32)

    def laser_callback(self, laser_data):
        # self.laser_info = np.array(laser_data.ranges, dtype=np.float32) # list
        
        self.laser_info = np.array(laser_data.ranges[511:571] + laser_data.ranges[571:1083] \
                            + laser_data.ranges[0:511], dtype=np.float32)
        
        regions_ = {
            'back':       min(min(msg.ranges[0:54] + msg.ranges[1026:1083]), 30), # 50 is the maximum value we can read
            'b-right1':   min(min(msg.ranges[54:162]), 30), # 50 because LiDAR specs is 50m. 
            'b-right2':   min(min(msg.ranges[162:270]), 30),
            'f-right2':   min(min(msg.ranges[270:378]), 30),
            'f-right1':   min(min(msg.ranges[378:486]), 30),
            'front':      min(min(msg.ranges[486:594]), 30),
            'f-left1':    min(min(msg.ranges[594:702]), 30),
            'f-left2':    min(min(msg.ranges[702:810]), 30),
            'b-left2':    min(min(msg.ranges[810:918]), 30),
            'b-left1':    min(min(msg.ranges[918:1026]), 30)
        }
        

    def cam_callback(self, camera_data):
        self.depth_info = np.array(camera_data.data, dtype=np.float32)

    def contact_callback(self, contact_data):
        contact_dict = {}
        for contact in contact_data.sensor:
            contact_dict[contact.name] = contact.force_torque[-1]
            if contact_dict[contact.name] >= 100: # seemingly suitable scale when observing the force data during simulation
                contact_dict[contact.name] = 5
            else:
                contact_dict[contact.name] = contact_dict[contact.name] / 100 * 5 # allow 0.3kg of knocking
        self.contact_info = np.array([contact_dict['front_plate'], contact_dict['front_left_plate'], contact_dict['back_left_plate'], contact_dict['back_plate'], contact_dict['back_right_plate'], contact_dict['front_right_plate']], dtype=np.float32)
    '''

    def robot_specific_reset(self, bullet_client, robot_loadedURDF):
        self.robot = robot_loadedURDF
        self._p = bullet_client
        OmniBase.robot_specific_reset(self, bullet_client)
        ### To initialise plugins for sensors in Pybullet ###
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        rev_joint_index_name_dic, fixed_joint_index_name_dic, link_names_to_ids_dic = self.get_properties()
        self.lidar = laserScannerModular(self._p, self.robot,                      
                    rev_joints=rev_joint_index_name_dic,
                    fixed_joints=fixed_joint_index_name_dic,
                    link_ids=link_names_to_ids_dic)
        self.contact_sensor_link_name_index_dic = {} # key: link_name, value: link_id or index
        # get pybullet laser link id from its name
        self.sensor_frame_id = rospy.get_param('~contact_sensor/sensor_id', None) # sensor reference frame, has to be an existing joint
        for sensor_id in self.sensor_frame_id:
            self.contact_sensor_link_name_index_dic[sensor_id] = link_names_to_ids_dic[sensor_id]
        ### To initialise plugins for sensors in Pybullet ###
        self.set_initial_orientation(yaw_center=0, yaw_random_spread=np.pi)
        if len(self.goal_set) != 0:
            ### Specific goal change
            if self.goal_change == 1:
                self.goal_count += 1
                self.goal_change = 0
                self.goal_repeat = 0
                if len(self.goal_set) == self.goal_count:
                    self.goal_count = 0
                    # self.goal_random_disp = random.choice([-0.5, 0.5]) # This is if the goal_set isn't fix. 
            ### Repetition goal change
            if self.goal_repeat == 3:
                self.goal_repeat = 0
                self.goal_count += random.choice([-1,1])
                if len(self.goal_set) <= self.goal_count or -len(self.goal_set) > self.goal_count:
                    self.goal_count = 0
            self.goal_x = self.goal_set[self.goal_count][0] # + self.goal_random_disp
            self.goal_y = self.goal_set[self.goal_count][1] # + self.goal_random_disp
            rospy.loginfo("Goal (x, y): (%s, %s)\n", str(self.goal_x), str(self.goal_y))
            self.goal_repeat += 1
            ###
        else:
            ### Random goal scenario
            try: # if the next goal has been set
                goal_message = rospy.wait_for_message('/goal', Pose, timeout=0.5)
                self.goal_x = goal_message.pose.x
                self.goal_y = goal_message.pose.y 
                rospy.loginfo("Goal for new episode has been set.")
            except: 
                self.goal_y = random.randint(-10,10) # check if this works. This is for changing the final goal position to achieve for next episode. 
                self.goal_x = random.randint(-10,10)
                self.goal_x += 0.5
                self.goal_y += 0.5
                rospy.loginfo("Randomising goal for new episode.")
                rospy.loginfo("Goal (x, y): (%s, %s)\n", str(self.goal_x), str(self.goal_y))
            ###

    def set_initial_orientation(self, yaw_center, yaw_random_spread):
        if not self.random_yaw:
            yaw = yaw_center
        else:
            yaw = yaw_center + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)

        position = [self.start_pos_x, self.start_pos_y, self.start_pos_z + 0.16]
        orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
        self.robot_body.reset_pose(position, p.getQuaternionFromEuler(orientation))
        self.initial_z = 1.5 # so the robot does not spawn in the ground? 

    ##### ACTION FUNCTIONS #####
    def apply_action(self, a):
        assert (np.isfinite(a).all())
        # Used variables in geometry_msgs/Twist
        # Vx=vel.linear.x;
        # Vy=vel.linear.y;
        # w=vel.angular.z; 

        # geometry_msgs/Vector3 linear
        # geometry_msgs/Vector3 angular

        # Vector3 (float): x, y, z
        vel_cmd = Twist()
        
        # vel_cmd.linear.x = self.action_set['vel_x'][a[0]]
        # vel_cmd.linear.y = self.action_set['vel_y'][a[1]]
        # vel_cmd.angular.z = self.action_set['vel_th'][a[2]]

        vel_cmd.linear.x = a[0]
        vel_cmd.linear.y = a[1]
        vel_cmd.angular.z = a[2]

        ### Threshold, clipping functions
        vel_cmd.linear.x = np.clip(vel_cmd.linear.x, self.min_vel_x, self.max_vel_x)
        vel_cmd.linear.y = np.clip(vel_cmd.linear.y, self.min_vel_y, self.max_vel_y)
        vel_cmd.angular.z = np.clip(vel_cmd.angular.z, self.min_vel_th, self.max_vel_th)

        if np.sqrt(pow(vel_cmd.linear.x,2)+pow(vel_cmd.linear.y,2)+pow(vel_cmd.angular.z,2)) > 0.51:
            vel_cmd.linear.x = vel_cmd.linear.x / self.max_vel_x * self.max_vel_all / 5 * 6
            vel_cmd.linear.y = vel_cmd.linear.y / self.max_vel_y * self.max_vel_all / 5 * 4
            vel_cmd.angular.z = vel_cmd.angular.z / self.max_vel_th * self.max_vel_all

        ### Deadzone velocity (negligible)
        if -0.05 < vel_cmd.linear.x < 0.05:
            vel_cmd.linear.x = 0
        if -0.05 < vel_cmd.linear.y < 0.05:
            vel_cmd.linear.y = 0
        if -0.05 < vel_cmd.angular.z < 0.05:
            vel_cmd.angular.z = 0

        self.cmd_vel_info[0] = vel_cmd.linear.x
        self.cmd_vel_info[1] = vel_cmd.linear.y
        self.cmd_vel_info[2] = vel_cmd.angular.z
        lin_vec = Vector3Stamped()
        lin_vec.header.frame_id = 'base_link'
        lin_vec.vector.x = vel_cmd.linear.x
        lin_vec.vector.y = vel_cmd.linear.y
        lin_vec.vector.z = 0 # self.cmd_vel_msg.linear.z
        ang_vec = Vector3Stamped()
        ang_vec.header.frame_id = 'base_link'
        ang_vec.vector.x = 0 # self.cmd_vel_msg.angular.x
        ang_vec.vector.y = 0 # self.cmd_vel_msg.angular.y
        ang_vec.vector.z = vel_cmd.angular.z
        lin_cmd_vel_in_odom = self.transformVector3('odom', lin_vec)
        ang_cmd_vel_in_odom = self.transformVector3('odom', ang_vec)
        # set vel directly on robot model
        self._p.resetBaseVelocity(self.robot, [lin_cmd_vel_in_odom.vector.x, lin_cmd_vel_in_odom.vector.y, lin_cmd_vel_in_odom.vector.z],
                                  [ang_cmd_vel_in_odom.vector.x, ang_cmd_vel_in_odom.vector.y, ang_cmd_vel_in_odom.vector.z])

        # self.vel_pub.publish(vel_cmd)

    def translation_matrix(self, direction):
        """copied from tf (listener.py)"""
        M = np.identity(4)
        M[:3, 3] = direction[:3]
        return M

    def quaternion_matrix(self, quaternion):
        """copied from tf (listener.py)"""
        epsilon = np.finfo(float).eps * 4.0
        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < epsilon:
            return np.identity(4)
        q *= math.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
            (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
            (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
            (                0.0,                 0.0,                 0.0, 1.0)
            ), dtype=np.float64)

    def fromTranslationRotation(self, translation, rotation):
        """copied from tf (listener.py)"""
        return np.dot(self.translation_matrix(translation), self.quaternion_matrix(rotation))

    def lookupTransform(self, target_frame='base_link', source_frame='odom'):
        """
        copied from tf (listener.py)
        source_frame = odom, world_frame, target_frame = base_link, robot_frame
        """
        # get robot pose from pybullet
        t, r = self._p.getBasePositionAndOrientation(self.robot)
        return [t[0], t[1], t[2]], [r[0], r[1], r[2], r[3]]

    def asMatrix(self, target_frame, hdr):
        """copied from tf (listener.py)"""
        translation,rotation = self.lookupTransform(target_frame, hdr.frame_id)
        return self.fromTranslationRotation(translation, rotation)

    def transformVector3(self, target_frame, v3s):
        """copied from tf (listener.py)"""
        mat44 = self.asMatrix(target_frame, v3s.header)
        mat44[0,3] = 0.0
        mat44[1,3] = 0.0
        mat44[2,3] = 0.0
        xyz = tuple(np.dot(mat44, np.array([v3s.vector.x, v3s.vector.y, v3s.vector.z, 1.0])))[:3]
        r = Vector3Stamped()
        r.header.stamp = v3s.header.stamp
        r.header.frame_id = target_frame
        r.vector = Vector3(*xyz)
        return r
    ##### ACTION FUNCTIONS #####

    def calc_state(self):
        ### tabulate Odom 'Callback'
        [self.odom_info[0],\
         self.odom_info[1],\
         self.odom_info[2]],\
        [self.odom_info[3],\
         self.odom_info[4],\
         self.odom_info[5],\
        self.odom_info[6]] = self._p.getBasePositionAndOrientation(self.robot)
        [self.odom_info[7],\
         self.odom_info[8],\
         self.odom_info[9]],\
        [self.odom_info[10],\
         self.odom_info[11],\
         self.odom_info[12]] = self._p.getBaseVelocity(self.robot)
        ###

        ### tabulate Laser 'Callback'
        self.laser_info = np.array(self.lidar.execute(), dtype=np.float32)
        
        if not self.laser_info.any():
            self.laser_info = np.full((1083,), 31.0, dtype=np.float32) # 31 points so that the number of output is same as that of laserscan merger
        # self.laser_info = np.array(np.concatenate([self.laser_info[511:571]] + [self.laser_info[571:1083]] \
        #                     + [self.laser_info[0:511]]), dtype=np.float32)
        # Direction has been reinstated to start from -ve 10 deg anticlockwise. 
        self.laser_info = np.array(np.concatenate([self.laser_info[1053:1083]] \
                            + [self.laser_info[0:1053]]), dtype=np.float32) # begin from the back
        ###

        ### tabulate Cam 'Callback'

        ###

        ### tabulate Contact 'Callback'
        contact_dict = {}
        for link_name in self.contact_sensor_link_name_index_dic:
            link_contact_msg = []
            # get link contact points from pybullet, link_contact_state will contain a list of contact points, each contact point contain a list of data. 
            link_contact_state = self._p.getContactPoints(linkIndexA=self.contact_sensor_link_name_index_dic[link_name])
            # check if there is contact point at all
            for contact_point in range(len(link_contact_state)):
                if link_contact_state[contact_point]:
                    link_contact_msg.append(link_contact_state[contact_point][8])
                    link_contact_msg.append(link_contact_state[contact_point][9])
            # processing to keep only 1 most high force contact point info
            if len(link_contact_msg) == 0:
                link_contact_msg_final = [0.0,0.0]
            if len(link_contact_msg) > 2:
                separating_d = link_contact_msg[0::2]
                normal_f = link_contact_msg[1::2]
                link_contact_msg_final = [min(separating_d),max(normal_f)] # only contain 1 pair of data
            if len(link_contact_msg) == 2:
                link_contact_msg_final = link_contact_msg
            contact_dict[link_name] = link_contact_msg_final[-1]
            if contact_dict[link_name] >= 100:
                contact_dict[link_name] = 5
            else:
                contact_dict[link_name] = contact_dict[link_name] / 100 * 5
        self.contact_info = np.array([contact_dict['front_plate'], contact_dict['front_left_plate'], contact_dict['back_left_plate'], contact_dict['back_plate'], contact_dict['back_right_plate'], contact_dict['front_right_plate']], dtype=np.float32)
        ###

        self.goal_theta = np.arctan2(self.goal_y - self.odom_info[1],self.goal_x - self.odom_info[0])
        self.goal_dist = np.linalg.norm([self.goal_y - self.odom_info[1], self.goal_x - self.odom_info[0]])
        # angle_to_goal = self.goal_theta - self.odom_info[5]
        robot_angle_wrt_origin = Quaternion(self.odom_info[6],self.odom_info[3],self.odom_info[4],self.odom_info[5]).to_euler()[2]
        angle_to_goal = self.goal_theta - robot_angle_wrt_origin
       
        if angle_to_goal > (np.pi):
            angle_to_goal = (-2)*np.pi + angle_to_goal
        if angle_to_goal < (-np.pi):
            angle_to_goal = 2*np.pi + angle_to_goal
        
        goal_info = np.array([self.goal_dist, angle_to_goal], dtype=np.float32)
        laser_processed_info = np.delete(self.laser_info, -1)
        laser_processed_info = np.delete(laser_processed_info, -1)
        laser_processed_info = np.delete(laser_processed_info, -1)
        # minimum pooling to acquire 18 readings from laser scan. Every 20 degrees a reading. 
        laser_processed_info = block_reduce(laser_processed_info, block_size=(60,), func=np.min)
        # return the list of observations, the len of this corresponds to the observation_dimensions
        return np.concatenate([goal_info] + [self.cmd_vel_info] + [self.contact_info] + [laser_processed_info]) # + [self.depth_info])
        # corresponds to state space of 2 goals, 3 cmd_vel, 6 contact, 18 laser scan. 
        # total of 29 state spaces

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        try:
            debugmode = 0
            if debugmode:
                print("calc_potential: self.goal_dist")
                print(self.goal_dist)
                print("self.scene.dt")
                print(self.scene.dt) # dt = timestep * frame_skip
                print("self.scene.frame_skip")
                print(self.scene.frame_skip)
                print("self.scene.timestep")
                print(self.scene.timestep)
            return - self.goal_dist / self.scene.dt
        except AttributeError:
            return - self.goal_dist
        return 1




class laserScannerModular:
    def __init__(self, pybullet, robot, **kargs):
        # get "import pybullet as _p" and store in self._p
        self._p = pybullet
        # get robot from parent class
        self.robot = robot
        # laser params
        self.number_of_lidars = rospy.get_param('~number_of_lidars', 0)
        self.lidar_topics = rospy.get_param('~lidar_topics', None) # scan1, scan2
        if (self.number_of_lidars != len(self.lidar_topics)):
          rospy.logwarn('Number of lidars do not match number of lidar topics, please check input parameters in yaml file.')
        self.laser_param_dic = {}
        ''' laser_param_dic = {scan1: {laser_frame_id: str, 
                                        _p_laser_link_id: int,
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
            # print('\n')
            # print(self.laser_param_dic[topic]['laser_frame_id'])
            # print("###")
            # print(link_names_to_ids_dic)
            # print('\n')
            if not self.laser_param_dic[topic]['laser_frame_id'] in link_names_to_ids_dic:
                rospy.logerr('laser reference frame "{}" not found in URDF model, cannot continue'.format(self.laser_param_dic[topic]['laser_frame_id']))
                rospy.logwarn('Available frames are: {}'.format(link_names_to_ids_dic))
                rospy.signal_shutdown('required param frame id not set properly')
                return
            self.laser_param_dic[topic]['_p_laser_link_id'] = link_names_to_ids_dic[self.laser_param_dic[topic]['laser_frame_id']]

            # create laser msg placeholder for publication
            self.laser_param_dic[topic]['laser_msg'] = LaserScan()
            # laser field of view
            angle_min = rospy.get_param('~laser_%s/angle_min' %topic[-1], -1.5707963) 
            angle_max = rospy.get_param('~laser_%s/angle_max' %topic[-1], 1.5707963)
            assert(angle_max > angle_min)
            self.laser_param_dic[topic]['numRays'] = rospy.get_param('~laser_%s/num_beams' %topic[-1], 50) # should be 512 beams but simulation becomes slow
            self.laser_param_dic[topic]['laser_msg'].range_min = rospy.get_param('~laser_%s/range_min' %topic[-1], 0.01)
            self.laser_param_dic[topic]['laser_msg'].range_max = rospy.get_param('~laser_%s/range_max' %topic[-1], 30.0)
            self.laser_param_dic[topic]['beam_visualisation'] = rospy.get_param('~laser_%s/beam_visualisation' %topic[-1], False)
            self.laser_param_dic[topic]['laser_msg'].angle_min = angle_min
            self.laser_param_dic[topic]['laser_msg'].angle_max = angle_max
            self.laser_param_dic[topic]['laser_msg'].angle_increment = (angle_max - angle_min) / self.laser_param_dic[topic]['numRays']
            # register this node in the network as a publisher in /scan topic
            # self.laser_param_dic[topic]['pub_laser_scanner'] = rospy.Publisher('scan%s' %topic[-1], LaserScan, queue_size=1)
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
        rm = self._p.getMatrixFromQuaternion(laser_orientation)
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
                    self._p.removeAllUserDebugItems()
                # get laser link position
                laser_state = self._p.getLinkState(self.robot, self.laser_param_dic[topic]['_p_laser_link_id'])
                # transform start and end position of the rays which were generated considering laser at the origin
                rayFrom, rayTo  = self.transform_rays(laser_state[0], laser_state[1], topic) # position + orientation
                # raycast using 4 threads
                results = self._p.rayTestBatch(rayFrom, rayTo, 4)
                for i in range(self.laser_param_dic[topic]['numRays']):
                    if self.laser_param_dic[topic]['beam_visualisation']:
                        hitObjectUid = results[i][0]
                        if hitObjectUid < 0:
                            # draw a line on pybullet gui for debug purposes in green because it did not hit any obstacle
                            self._p.addUserDebugLine(rayFrom[i], rayTo[i], self.rayMissColor)
                        else:
                            # draw a line on pybullet gui for debug purposes in red because it hited obstacle, results[i][3] -> hitPosition
                            self._p.addUserDebugLine(rayFrom[i], results[i][3], self.rayHitColor)
                    # compute laser ranges from hitFraction -> results[i][2]
                    self.laser_param_dic[topic]['laser_msg'].ranges.append(results[i][2] * self.laser_param_dic[topic]['laser_msg'].range_max)
                # update laser time stamp with current time
                self.laser_param_dic[topic]['laser_msg'].header.stamp = rospy.Time.now()
                # publish scan
                # self.laser_param_dic[topic]['pub_laser_scanner'].publish(self.laser_param_dic[topic]['laser_msg'])
        return self.laser_param_dic[topic]['laser_msg'].ranges