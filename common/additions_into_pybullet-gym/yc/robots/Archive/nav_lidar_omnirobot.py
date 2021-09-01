from pybulletgym.envs.yc.robots.omni_base import OmniBase
from pybulletgym.envs.yc.robots.robot_bases import URDFBasedRobot # like atlas example, with omn_descriptions in apybulletgym/envs/assets/robot
from pybulletgym.envs.yc.robots.robot_bases import MJCFBasedRobot # like other examples, with .xml file in apybulletgym/envs/assets/mjcf
import numpy as np
import pybullet as p
from geometry_msgs.msg import Twist, Vector3Stamped, Pose, Vector3, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, Image
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from ros_pybullet_rl.msg import DepthImageProcessed
import rospy
import random
from squaternion import Quaternion
from skimage.measure import block_reduce
from math import sqrt

from time import sleep 

from std_srvs.srv import Empty
import roslaunch

import os
from signal import signal, SIGINT

class NavLidarOmnirobot(OmniBase, URDFBasedRobot):
    random_yaw = False
    # Node ID: 1 - front left wheels, 2 - back, 3 - front right. 
    foot_list = ["front_left_wheel", "back_wheel","front_right_wheel"] # this is the link name of the locomotor in contact with the ground. To be tracked. 
    # rename foot_list to the 3 wheels link name in contact with the ground of the Omnirobot. 
    def __init__(self):
        start_pos_x = 0.0
        OmniBase.__init__(self, goal_x=0, goal_y=0, start_pos_x=start_pos_x)
        URDFBasedRobot.__init__(self, "omnirobot_v3/urdf/omnirobot_v3.urdf",
                                "Omnirobot", 
                                action_dim=3, 
                                obs_dim=26# original 29, 23 is exclude contact_data
                                )

        # Initialise variables
        self.odom_info = np.zeros((13,), dtype=np.float32) # list of 13 floats. 
        self.laser_info = np.full((1083,), 31.0, dtype=np.float32) # was list of 1083 floats. 
        self.depth_info = np.zeros((307200,), dtype=np.float32) # list of 307200 floats.
        self.contact_info = np.zeros((6,), dtype=np.float32) # list of 6 floats. !!! TAKE NOTE THE ORDER OF THIS IS IMPORTANT IN GENERALISING PROGRAM !!! 
        self.cmd_vel_info = np.zeros((3,), dtype=np.float32) # list of 3 floats. 
        self.nav_vel_info = np.zeros((3,), dtype=np.float32)
        self.nav_cmd_vel_info = np.zeros((3,), dtype=np.float32)

        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.vis_pub = rospy.Publisher('/marker/cmd_vel', Marker, queue_size=1)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.laser_sub = rospy.Subscriber('scan', LaserScan, self.laser_callback)
        self.cam_sub = rospy.Subscriber('d_image_processed', DepthImageProcessed, self.cam_callback)
        self.nav_vel_sub = rospy.Subscriber('nav/cmd_vel', Twist, self.nav_vel_callback)
        # self.plan_sub = rospy.Subscriber('move_base/NavfnROS/plan', Path, self.path_callback)

        self.ori_lidar_pub = rospy.Publisher('/original_lidar',LaserScan, queue_size=5)

        # For specific goal change
        self.goal_x = 0
        self.goal_y = 0
        self.goal_change = 0
        self.goal_count = 0
        self.goal_repeat = 0
        self.goal_random_disp = 0
        self.nav_msg_received = 0

        self.select_env = rospy.get_param("~select_env")
        self.is_validate = rospy.get_param("~is_validate")
        if not self.is_validate:
            self.goal_set = eval(rospy.get_param("~env_obj{}/goal_set".format(self.select_env)))
        else: 
            self.goal_set = eval(rospy.get_param("~env_obj{}/validation_goal_set".format(self.select_env)))

        # APF params
        self.kf_lower = 0.4 # rospy.get_param("~kf_lower", 0.4) # in proximity filter, value = 1.0

        self.count = 0
        self.max_vel_x = rospy.get_param("~robot/max_vel_x", 0.51)
        self.max_vel_y = rospy.get_param("~robot/max_vel_y", 0.46)
        self.max_vel_th = rospy.get_param("~robot/max_vel_th", 1.00)
        self.min_vel_x = rospy.get_param("~robot/min_vel_x", -0.51)
        self.min_vel_y = rospy.get_param("~robot/min_vel_y", -0.46)
        self.min_vel_th = rospy.get_param("~robot/min_vel_th", -1.00)
        self.max_vel_all = rospy.get_param("~robot/max_vel_all", 0.2655)
        # self.original_laser = LaserScan()

        ### Initialise navigation services
        self.main_dir = rospy.get_param('~main_dir')
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.clear_costmap = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)

    def nav_vel_callback(self, nav_vel_data):
        # Set to publish at 10 Hz. 
        self.nav_vel_info = np.array([nav_vel_data.linear.x,nav_vel_data.linear.y,nav_vel_data.angular.z], dtype=np.float32)
        self.nav_msg_received = 1
    ### Cannot feed the path in because the path is global path, has nothing to do with local path. 
    
    # def path_callback(self, path_data):
    #     print("Length of path is: ", len(path_data.poses))
    #     print("Type of data is: ", type(path_data.poses))
    #     self.path_info = np.array(path_data.poses)


    def odom_callback(self, odom_data):
        self.odom_info = np.array([odom_data.pose.pose.position.x,odom_data.pose.pose.position.y,\
            odom_data.pose.pose.position.z,odom_data.pose.pose.orientation.x,\
            odom_data.pose.pose.orientation.y,odom_data.pose.pose.orientation.z,\
            odom_data.pose.pose.orientation.w,odom_data.twist.twist.linear.x,\
            odom_data.twist.twist.linear.y,odom_data.twist.twist.linear.z,\
            odom_data.twist.twist.angular.x,odom_data.twist.twist.angular.y,\
            odom_data.twist.twist.angular.z], dtype=np.float32)

    def laser_callback(self, laser_data):
        self.laser_info = np.array(laser_data.ranges[1053:1083] \
                            + laser_data.ranges[0:1053], dtype=np.float32) # begin from the back
        '''
        regions_ = {
            'back':       min(min(msg.ranges[0:54] + msg.ranges[1026:1083]), 30), # 30 is the maximum value we can read
            'b-right1':   min(min(msg.ranges[54:162]), 30), # 30 because LiDAR specs is 30m. 
            'b-right2':   min(min(msg.ranges[162:270]), 30),
            'f-right2':   min(min(msg.ranges[270:378]), 30),
            'f-right1':   min(min(msg.ranges[378:486]), 30),
            'front':      min(min(msg.ranges[486:594]), 30),
            'f-left1':    min(min(msg.ranges[594:702]), 30),
            'f-left2':    min(min(msg.ranges[702:810]), 30),
            'b-left2':    min(min(msg.ranges[810:918]), 30),
            'b-left1':    min(min(msg.ranges[918:1026]), 30)
        }
        '''
        # self.original_laser = laser_data

    def cam_callback(self, camera_data):
        self.depth_info = np.array(camera_data.data, dtype=np.float32)

    def robot_specific_reset(self, bullet_client):
        ### Relaunch move_base node to avoid clear costmap rosservice not-working error. 
        try:
            self.move_base_launch.shutdown()
        except:
            pass
        self.launch_move_base_node()
        ###

        OmniBase.robot_specific_reset(self, bullet_client)
        
        ''' ### Do not need this section anymore if relaunching move_base node in every episode
        if self.goal_x == 0 and self.goal_y == 0:
            pass
        else: ### Clear costmap rosservice is removed because of a bug where clear_costmaps does not work after being called several number of times. 
            # rospy.wait_for_service('/move_base/clear_costmaps')
            self.clear_costmap() # to solve the problem where robot respawns into the map, and the map may get messed up
            # print("\nCostmap cleared.\n")
        '''

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

    def launch_move_base_node(self):
        ### Initialise navigation service
        # self.move_base_node = roslaunch.core.Node("move_base", "move_base")
        # self.move_base_launch = roslaunch.scriptapi.ROSLaunch()
        self.move_base_launch = roslaunch.parent.ROSLaunchParent(self.uuid, ["{}/launch/navigation/move_base_eband.launch".format(self.main_dir)])
        self.move_base_launch.start()
        # self.move_base_launch.launch(self.move_base_node)

    def publish_goal(self):
        local_goal = PoseStamped()
        local_goal.header.stamp = rospy.Time.now()
        local_goal.header.frame_id = "odom"
        local_goal.pose.position.x = self.goal_x
        local_goal.pose.position.y = self.goal_y
        local_goal.pose.position.z = 0
        local_goal.pose.orientation.x = 0
        local_goal.pose.orientation.y = 0
        local_goal.pose.orientation.z = 0
        local_goal.pose.orientation.w = 1
        self.goal_pub.publish(local_goal)

    def set_initial_orientation(self, yaw_center, yaw_random_spread):
        if not self.random_yaw:
            yaw = yaw_center
        else:
            yaw = yaw_center + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)

        position = [self.start_pos_x, self.start_pos_y, self.start_pos_z + 0.16]
        orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
        self.robot_body.reset_pose(position, p.getQuaternionFromEuler(orientation))
        self.initial_z = 1.5 # so the robot does not spawn in the ground? 

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

        vel_cmd.linear.x = a[0] + self.nav_cmd_vel_info[0]
        vel_cmd.linear.y = a[1] + self.nav_cmd_vel_info[1]
        vel_cmd.angular.z = a[2] + self.nav_cmd_vel_info[2]

        ''' EBand Planner velocity limits
        max_vel_lin: 0.5 # 0.2
        max_vel_th: 1.0 # 0.1
        min_vel_lin: 0.05 # 0.1
        min_vel_th: -1.0 # 0.0
        '''

        ### Threshold, clipping functions
        vel_cmd.linear.x = np.clip(vel_cmd.linear.x, self.min_vel_x, self.max_vel_x)
        vel_cmd.linear.y = np.clip(vel_cmd.linear.y, self.min_vel_y, self.max_vel_y)
        vel_cmd.angular.z = np.clip(vel_cmd.angular.z, self.min_vel_th, self.max_vel_th)

        if sqrt(pow(vel_cmd.linear.x,2)+pow(vel_cmd.linear.y,2)+pow(vel_cmd.angular.z,2)) > 0.51:
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
        self.vel_pub.publish(vel_cmd)
        self.visualise_vel(vel_cmd)

    def visualise_vel(self, vel_cmd):
        # visualise cmd_vel 
        marker = Marker()
        temp = Point()
        marker.header.frame_id = "base_link";
        marker.header.stamp = rospy.Time.now()
        marker.ns = "cmd_vel"
        marker.id = 0
        marker.type = marker.ARROW
        marker.action = marker.ADD
        # Display BLUE ARROW from current position to direction of travel
        temp.x, temp.y, temp.z = 0, 0, 1
        marker.points.append(temp) # current position wrt base_link
        temp.x, temp.y, temp.z = vel_cmd.linear.x, vel_cmd.linear.y, 1
        marker.points.append(temp)
        marker.scale = Vector3(x=0.1,y=0.2,z=0.2)
        marker.color = ColorRGBA(b=1.0,a=1.0)
        marker.lifetime = rospy.Duration(0.5)
        self.vis_pub.publish(marker)        

    def calc_state(self):
        ### Environments begin with this
        self.contact_delta_v = Twist()
        if self.nav_msg_received == 0:
            self.publish_goal() # solves the problem whereby the first episode does not have a plan because the move_base node initialises too slow to receive the goal_node. 
        self.goal_theta = np.arctan2(self.goal_y - self.odom_info[1],self.goal_x - self.odom_info[0])
        self.goal_dist = np.linalg.norm([self.goal_y - self.odom_info[1], self.goal_x - self.odom_info[0]])
        robot_angle_wrt_origin = Quaternion(self.odom_info[6],self.odom_info[3],self.odom_info[4],self.odom_info[5]).to_euler()[2]
        angle_to_goal = self.goal_theta - robot_angle_wrt_origin # - odom_info[5]

        if angle_to_goal > (np.pi):
            angle_to_goal = (-2)*np.pi + angle_to_goal
        if angle_to_goal < (-np.pi):
            angle_to_goal = 2*np.pi + angle_to_goal

        ### EBAND velocity
        if (self.nav_cmd_vel_info == self.nav_vel_info).all(): # This is always varying, never be the same unless nav_cmd_vel is not published. 
            self.nav_msg_received = 0            
        self.nav_cmd_vel_info = self.nav_vel_info

        goal_info = np.array([self.goal_dist, angle_to_goal], dtype=np.float32)
        laser_processed_info = np.delete(self.laser_info, [-3,-2,-1])
        # laser_processed_info = np.delete(laser_processed_info, -1)
        # laser_processed_info = np.delete(laser_processed_info, -1)
        # minimum pooling to acquire 18 readings from laser scan. Every 20 degrees a reading. 
        laser_processed_info = block_reduce(laser_processed_info, block_size=(60,), func=np.min)
        # return the list of observations, the len of this corresponds to the observation_dimensions
        ### Apply Gaussian noise to laser data
        gaussian_noise = np.random.normal(0, 0.3, 18).astype('float32')
        laser_processed_info += gaussian_noise

        ### This is to check the positioning of the LiDAR data points. 
        # self.original_laser.ranges = laser_processed_info # [0:5]
        # self.original_laser.angle_increment = 0.349
        # self.ori_lidar_pub.publish(self.original_laser)

        return np.concatenate([goal_info] + [self.nav_cmd_vel_info] + [self.cmd_vel_info] + [laser_processed_info]) # + [self.laser_info]) + [self.depth_info])
        # corresponds to state space of dist, angle, 3 nav_cmd_vel, 3 prev cmd_vel, 6 contact, 18 laser scan. 
        # total of 32 state spaces

