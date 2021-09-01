from pybulletgym.envs.yc.envs.py_env_bases import BaseBulletEnv
from pybulletgym.envs.yc.scenes import StadiumScene
import numpy as np
import pybullet
from pybulletgym.envs.yc.robots import PyOmnirobot

from gym.envs.registration import register

import os
import importlib
import rospy

import pybullet_data

from std_srvs.srv import Empty

import random


class PyOmniBaseBulletEnv(BaseBulletEnv):
  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects

  def __init__(self, robot, render, goal_x, goal_y):
    print("OmniBase::__init__")
    BaseBulletEnv.__init__(self, robot, render)
    self.camera_x = 0
    self.goal_x = goal_x
    self.goal_y = goal_y
    self.stateId = -1
    self.reward_range = (-np.inf, np.inf)
    self.prev_goal_dist = 0.0
    self.prev_angle_to_goal = 0.0
    self.last_three_actions = []

    # Parameters to control sensitivity of laser and force torque sensors
    self.detect_radius = 0.5
    self.kp_linear = 0.02
    self.kf_lower = 0.4

    self._seed()

    self.declared_dyn_bodies = 0

    # Stop episode condition to avoid bad samples
    self.num_collision = 0

  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = StadiumScene(bullet_client,
                                      gravity=9.8,
                                      timestep=0.01, # 0.01 # timestep=0.0165 / 4,
                                      frame_skip=4)
    return self.stadium_scene


  def reset(self):
    if (self.stateId >= 0):
      self._p.restoreState(self.stateId)

    r = BaseBulletEnv._reset(self) # r = calc_state() # this eventually calls robot.reset, which is from Omnirobot class. 
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

    return r


  def _isDone(self, reach_goal):
    return reach_goal < 0.3


  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit
    done = self._isDone(state[0])
    #if not np.isfinite(state).all():
    #  print("~INF~", state)
    #  done = True
    if done:
      rospy.loginfo("Robot reached goal.\n")
      self.robot.goal_change = 1
      ###
    completion_reward = 10 if done else 0 # 5 # Not to set too high as well given that in every timestep within threshold of the goal will gain reward
    # done = 0 condition moved to after the distance_displacement_cost
    
    # timestep_cost = -self.stadium_scene.timestep * 0.1 # the longer it takes the worst. 

    current_angle_to_goal = state[1] # angle in radians 
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
      x_facing_goal_cost += -0.05
    if abs(current_angle_to_goal) > 1.0471:
      x_facing_goal_cost += -0.05
    if abs(current_angle_to_goal) > 1.5707:
      x_facing_goal_cost += -0.05
    if abs(current_angle_to_goal) > 2.0943:
      x_facing_goal_cost += -0.05
    if abs(current_angle_to_goal) > 2.6179:
      x_facing_goal_cost += -0.05
    self.prev_angle_to_goal = current_angle_to_goal

    proximity_cost = 0
    if min(state[11:29]) <= self.detect_radius:
      proximity_cost -= self.kp_linear * (self.detect_radius - min(state[14:32]))
    # There is no exponent, it's just a scaling factor. 
    # laser as close as 0.49, cost is 0.02 * (0.5 - 0.49)

    collision_cost = 0.0
    if max(state[5:11]) != 0:
      collision_cost -= self.kf_lower * max(state[8:14])
      if max(state[5:11]) >= 3.0:
        self.num_collision += 1
      if self.num_collision == 100:
        done = True # stop episode

    '''for contact in state[5:11]: # progressively higher the penalty the larger the contact force, was flat -0.2.
      if 0.0 < contact <= 1.0:
        collision_cost += -0.10
      if 1.0 < contact <= 2.0:
        collision_cost += -0.15
      if 2.0 < contact <= 3.0:
        collision_cost += -0.20
      if 3.0 < contact <= 4.0:
        collision_cost += -0.25
      if 4.0 < contact <= 5.0:
        collision_cost += -0.30'''
    if collision_cost != 0:
      rospy.logwarn("Robot experiencing collision!") 

    current_goal_dist = state[0] # Check if relative distance towards goal is shorter with every step. 
    displacement = current_goal_dist - self.prev_goal_dist
    if displacement >= 0.0:
      relative_distance_progress_cost = -0.05
    else: 
      relative_distance_progress_cost = 0.01
    self.prev_goal_dist = current_goal_dist
    ### if within threshold of the goal, do not penalise
    if done:
      relative_distance_progress_cost = 0
    # done = 0 # to teach the robot the knowledge to stay within a certain distance of the goal

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
        self.dynamic_obj[bod]['F_factor'] = 295 # 300 as the F_external hypotenuse that will make a 75 kg mass move at ~ constant velocity. 
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
    ###

    debugmode = 0
    if (debugmode):
      print("progress")
      print(progress)

    self.rewards = [
      completion_reward, collision_cost, relative_distance_progress_cost, proximity_cost, x_facing_goal_cost
    ]
    if (debugmode):
      print("rewards=")
      print(self.rewards)
      print("sum rewards")
      print(sum(self.rewards))
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)
    #self.count += 1
    #print("Step: ", self.count)
    #state = np.delete(state, 10)
    #state = np.delete(state, 9)
    #state = np.delete(state, 8)
    #state = np.delete(state, 7)
    #state = np.delete(state, 6)
    #state = np.delete(state, 5)
    # print(type(state))
    # print(state)
    # print("goal dist input: ", state[0], "\nvel data input: ", state[3], "\nlaser data: ", state[-1])
    # print("type - goal dist input: ", type(state[0]), "\nvel data input: ", type(state[3]), "\nlaser data: ", type(state[-1]))
    return state, sum(self.rewards), bool(done), {}

  def camera_adjust(self):
    x, y, z = self.robot.body_xyz # was self.robot.body_real_xyz, this variable is found in omni_bases.py, inherited by Omnirobot

    self.camera_x = x
    self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)


class PyOmniBulletEnv(PyOmniBaseBulletEnv):
  # as long as this class is called, it will have variables defined in OmniBaseBulletEnv as well.
  def __init__(self, render=False):
    self.goal_x = 10.0
    self.goal_y = 10.0
    self.robot = PyOmnirobot(self.goal_x,self.goal_y)
    # self.robot_loadedURDF_objects = self.robot.objects # Yes, this works, proven. This is to load plugins. Object from robot_bases.py
    PyOmniBaseBulletEnv.__init__(self, self.robot, render, self.goal_x, self.goal_y)
