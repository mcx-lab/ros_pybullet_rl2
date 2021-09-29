from pybulletgym.envs.yc.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.yc.scenes import StadiumScene
import numpy as np
import pybullet
from pybulletgym.envs.yc.robots import NavOmnirobot2

from gym.envs.registration import register

import os
import importlib
import rospy

import pybullet_data

from std_srvs.srv import Empty

import random


class NavOmniBase2BulletEnv(BaseBulletEnv):
  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects

  def __init__(self, robot, render):
    print("NavOmniBase2::__init__")
    BaseBulletEnv.__init__(self, robot, render)
    self.camera_x = 0
    self.stateId = -1
    self.reward_range = (-np.inf, np.inf)
    self.prev_goal_dist = 0.0
    self.prev_angle_to_goal = 0.0
    self.last_three_actions = []
    
    # Parameters to control sensitivity of laser and force torque sensors
    self.detect_radius = rospy.get_param("~detect_radius", 0.5)
    self.kp_linear = rospy.get_param("~kp_linear", 0.02)
    self.kf_lower = rospy.get_param("~kf_lower", 0.4) # in proximity filter, value = 1.0
    self.kf_move_dynamic_obj = rospy.get_param("~kf_move_dynamic_obj", 61)

    self._seed()

    self.rosWrapper = None
    self.declared_dyn_bodies = 0

    # Stop episode condition to avoid bad samples
    self.num_collision = 0


  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = StadiumScene(bullet_client,
                                      gravity=9.8,
                                      timestep=0.01,
                                      frame_skip=4)
    return self.stadium_scene


  def reset(self):
    if (self.stateId >= 0):
      self._p.restoreState(self.stateId)

    r, self.rosWrapper = BaseBulletEnv._reset(self) # r = calc_state() # this eventually calls robot.reset, which is from Omnirobot class. 
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
        self._p, self.stadium_scene.ground_plane_mjcf)
    self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                            self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
    self.dynamic_obj = {}
    self.declared_dyn_bodies = 0
    self.num_collision = 0

    return r # need to return r? This r refers to observation. 


  def _isDone(self, reach_goal):
    return reach_goal < 0.2


  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    for task in self.rosWrapper.plugins:
      task.execute()

    norm_state, state = self.robot.calc_state()  # also calculates self.joints_at_limit
    done = self._isDone(state[0][0])

    if done:
      rospy.loginfo("Robot reached goal.\n")
      self.robot.goal_change = 1
      ###
    completion_reward = 10 if done else 0 # 5 # Not to set too high as well given that in every timestep within threshold of the goal will gain reward

    '''if state[2][0] != 0 and state[2][1] != 0 and state[2][2] != 0: # If no action taken, smaller punishment
      movement_cost = 0 # movement_cost = -0.2 # consider maybe the abs|movement cost| should be larger than the |facing goal reward|
    else: # not moving, to some extent of timestep_cost, not moving is better. 
      movement_cost = 0 # -0.05'''

    current_angle_to_goal = state[0][1] # angle in radians 
    angle_displacement = abs(current_angle_to_goal) - abs(self.prev_angle_to_goal) # relative displacement towards goal 
    # if current angle to goal and prev angle are both same sign. 
    if angle_displacement >= 0:
      x_facing_goal_cost = -0.02
    else:
      x_facing_goal_cost = 0.0
    # if current angle to goal positive, prev angle negative, 
    if current_angle_to_goal > 1.5707 and self.prev_angle_to_goal < -1.5707:
      x_facing_goal_cost = -0.02
    # if current angle to goal negative, prev angle positive,
    if current_angle_to_goal < -1.5707 and self.prev_angle_to_goal > 1.5707:
      x_facing_goal_cost = -0.02

    if abs(current_angle_to_goal) <= 0.5235: # check if within 30 degrees of facing the goal
      x_facing_goal_cost = 0.0

    if abs(current_angle_to_goal) > 0.5235:
      x_facing_goal_cost -= abs(current_angle_to_goal) * 0.1
      # x_facing_goal_cost += -0.05
    self.prev_angle_to_goal = current_angle_to_goal

    proximity_cost = 0
    if min(state[4]) <= self.detect_radius:
      proximity_cost -= self.kp_linear * (self.detect_radius - min(state[4]))
    
    # There is no exponent, it's just a scaling factor. 
    # laser as close as 0.49, cost is 0.02 * (0.5 - 0.49)

    '''if np.arctan2(abs(state[2][1]),state[2][0]) > 0.8: # redundant: or np.arctan2(abs(state[3]),state[2]) < 0: # contain the meaning of state[2] < 0.
      x_facing_movement_cost = -0.025
    else:
      x_facing_movement_cost = 0.01 # reward'''

    collision_cost = 0.0
    if max(state[3]) != 0:
      collision_cost -= self.kf_lower * max(state[3])
      if max(state[3]) >= 3.0:
        self.num_collision += 1
      if self.num_collision == 100:
        done = True # stop episode
    # There is no exponent, it's just a scaling factor.

    if collision_cost != 0.0:
      rospy.logwarn("Robot experiencing collision!") 

    current_goal_dist = state[0][0] # Check if relative distance towards goal is shorter with every step. 
    displacement = current_goal_dist - self.prev_goal_dist
    if displacement > 0.0:
      relative_distance_progress_cost = -0.05
    elif displacement < 0.0: 
      relative_distance_progress_cost = 0.01
    else: 
      relative_distance_progress_cost = 0.0
    self.prev_goal_dist = current_goal_dist

    self.move_dynamic_obj()
    
    self.rewards = [
      completion_reward, proximity_cost, collision_cost, relative_distance_progress_cost, x_facing_goal_cost
    ]

    self.HUD(norm_state, a, done)
    self.reward += sum(self.rewards)

    return norm_state, sum(self.rewards), bool(done), {}


  def move_dynamic_obj(self):
    ### Introduce random motion to the dynamic bodies # Designed for 1000 timesteps. 
    if self.declared_dyn_bodies == 0:
      self.declared_dyn_bodies = 1
      self.dynamic_bodies = self.scene.dyn_bodies
      num_body = 0
      for bod in self.dynamic_bodies:
        self.dynamic_obj[bod] = {}
        self.dynamic_obj[bod]['id'] = num_body
        self.dynamic_obj[bod]['reinstate_direction'] = 1
        self.dynamic_obj[bod]['count'] = 0
        self.dynamic_obj[bod]['new_direction_interval'] = 100 * random.randint(2,9) # Can change to vary behaviour of dynamic obstacle
        self.dynamic_obj[bod]['time_for_rest'] = self.dynamic_obj[bod]['new_direction_interval'] - int((100 * random.uniform(-0.9,1.5))) # Can change to vary behaviour of dynamic obstacle
        self.dynamic_obj[bod]['fell'] = 0
        if len(self.scene.dyn_bodies_action_angle) != 0:
          self.dynamic_obj[bod]['action_angle_set'] = self.scene.dyn_bodies_action_angle[num_body]
          self.dynamic_obj[bod]['action_angle_set'] = [eval(angle) for angle in self.dynamic_obj[bod]['action_angle_set']]
          self.dynamic_obj[bod]['action_angle_index'] = 0
        else: 
          self.dynamic_obj[bod]['action_angle_set'] = [None] 
        num_body += 1
    for bod in self.dynamic_bodies: # same as: for bod in self.dynamic_obj
      if self.dynamic_obj[bod]['fell']:
        continue
      if self.dynamic_obj[bod]['reinstate_direction']:
        # print("\n\nRunning reinstate direction for: ", bod)
        if self.dynamic_obj[bod]['action_angle_set'][0] != None:
          self.dynamic_obj[bod]['F_angle'] = self.dynamic_obj[bod]['action_angle_set'][self.dynamic_obj[bod]['action_angle_index']]
          self.dynamic_obj[bod]['action_angle_index'] += 1
          if self.dynamic_obj[bod]['action_angle_index'] == len(self.dynamic_obj[bod]['action_angle_set']):
            self.dynamic_obj[bod]['action_angle_index'] = 0
        else:
          self.dynamic_obj[bod]['F_angle'] = random.uniform(-np.pi,np.pi)
        self.dynamic_obj[bod]['F_factor'] = self.kf_move_dynamic_obj # 300 as the F_external hypotenuse that will make a 75 kg mass move at ~ constant velocity. 
        self.dynamic_obj[bod]['F_ext'] = [np.cos(self.dynamic_obj[bod]['F_angle']) * self.dynamic_obj[bod]['F_factor'], np.sin(self.dynamic_obj[bod]['F_angle']) * self.dynamic_obj[bod]['F_factor'], 0] # For close to constant speed, 
        self.dynamic_obj[bod]['position'], orient = self._p.getBasePositionAndOrientation(bod)
        self.dynamic_obj[bod]['reinstate_direction'] = 0
      if self.dynamic_obj[bod]['count'] == self.dynamic_obj[bod]['time_for_rest']: # 
        self.dynamic_obj[bod]['F_ext'] = [0, 0, 0]
      if self.dynamic_obj[bod]['count'] == self.dynamic_obj[bod]['new_direction_interval']:
        self.dynamic_obj[bod]['reinstate_direction'] = 1
        self.dynamic_obj[bod]['count'] = 0
      if self._p.getContactPoints(bodyA=bod, linkIndexA=-1) == ():
        # print('Object fell.')
        self.dynamic_obj[bod]['fell'] = 1
      self._p.applyExternalForce(objectUniqueId=bod, linkIndex=-1, forceObj=self.dynamic_obj[bod]['F_ext'], posObj=(self.dynamic_obj[bod]['position'][0],self.dynamic_obj[bod]['position'][1],0.0), flags=self._p.WORLD_FRAME)
      self.dynamic_obj[bod]['count'] += 1

  def camera_adjust(self):
    x, y, z = self.robot.body_xyz # was self.robot.body_real_xyz, this variable is found in omni_bases.py, inherited by Omnirobot

    self.camera_x = x
    self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)


class NavOmni2BulletEnv(NavOmniBase2BulletEnv):
  # as long as this class is called, it will have variables defined in OmniBaseBulletEnv as well.
  def __init__(self, render=False):
    self.robot = NavOmnirobot2()
    NavOmniBase2BulletEnv.__init__(self, self.robot, render)