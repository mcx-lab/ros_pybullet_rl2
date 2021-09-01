import pybullet
import rospy, pybullet_data, importlib
import gym, gym.spaces, gym.utils
import numpy as np
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from ros_pybullet_rl2.utils.function_exec_manager import FuncExecManager
from threading import Thread
import time, sys, copy

class XmlBasedRobot:
    """
    Base class for mujoco .xml based agents.
    """

    self_collision = True

    def __init__(self, robot_name, action_dim, obs_dim, action_space, observation_space, self_collision):
        self.parts = None
        self.objects = []
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None

        self.action_space = action_space

        # state space
        self.observation_space = observation_space

        self.robot_name = robot_name
        self.self_collision = self_collision

    def addToScene(self, bullet_client, bodies):
        self._p = bullet_client

        if self.parts is not None:
            parts = self.parts
        else:
            parts = {}

        if self.jdict is not None:
            joints = self.jdict
        else:
            joints = {}

        if self.ordered_joints is not None:
            ordered_joints = self.ordered_joints
        else:
            ordered_joints = []

        if np.isscalar(bodies):  # streamline the case where bodies is actually just one body
            bodies = [bodies]

        dump = 0
        for i in range(len(bodies)):
            if self._p.getNumJoints(bodies[i]) == 0:
                part_name, robot_name = self._p.getBodyInfo(bodies[i])
                self.robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(self._p, part_name, bodies, i, -1)
            for j in range(self._p.getNumJoints(bodies[i])):
                self._p.setJointMotorControl2(bodies[i], j, pybullet.POSITION_CONTROL, positionGain=0.1, velocityGain=0.1, force=0)
                jointInfo = self._p.getJointInfo(bodies[i], j)
                joint_name=jointInfo[1]
                part_name=jointInfo[12]

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                if dump: print("ROBOT PART '%s'" % part_name)
                if dump: print("ROBOT JOINT '%s'" % joint_name)  # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )

                parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

                if part_name == self.robot_name:
                    self.robot_body = parts[part_name]

                if i == 0 and j == 0 and self.robot_body is None:  # if nothing else works, we take this as robot_body
                    parts[self.robot_name] = BodyPart(self._p, self.robot_name, bodies, 0, -1)
                    self.robot_body = parts[self.robot_name]

                if joint_name[:6] == "ignore":
                    Joint(self._p, joint_name, bodies, i, j).disable_motor()
                    continue

                if joint_name[:8] != "jointfix":
                    joints[joint_name] = Joint(self._p, joint_name, bodies, i, j)
                    ordered_joints.append(joints[joint_name])

                    joints[joint_name].power_coef = 100.0

        return parts, joints, ordered_joints, self.robot_body

    def robot_specific_reset(self, physicsClient): # this won't be called since it is robot_specific_reset, possibly likely to be different for different robots. 
        pass

    def reset_pose(self, position, orientation):
        self.parts[self.robot_name].reset_pose(position, orientation)


class MJCFBasedRobot(XmlBasedRobot):
    """
    Base class for mujoco .xml based agents.
    """

    def __init__(self, model_xml, robot_name, action_dim, obs_dim, self_collision=True):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)
        self.model_xml = model_xml
        self.doneLoading = 0

    def reset(self, bullet_client):
        full_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "mjcf", self.model_xml)

        self._p = bullet_client
        if self.doneLoading == 0:
            self.ordered_joints = []
            self.doneLoading=1
            if self.self_collision:
                self.objects = self._p.loadMJCF(full_path, flags=pybullet.URDF_USE_SELF_COLLISION|pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects    )
            else:
                self.objects = self._p.loadMJCF(full_path)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)
        self.robot_specific_reset(self._p)

        norm_state, s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return norm_state

    @staticmethod
    def calc_potential():
        return 0


class URDFBasedRobot(XmlBasedRobot):
    """
    Base class for URDF .xml based robots.
    """

    def __init__(self, model_urdf, robot_name, action_dim, obs_dim, action_space, observation_space, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], fixed_base=False, self_collision=False):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, action_space, observation_space, self_collision)

        self.model_urdf = model_urdf
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.fixed_base = fixed_base
        self.episode = 0

    def reset(self, bullet_client):
        self._p = bullet_client
        self.ordered_joints = []

        full_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "robots", self.model_urdf)
        if self.objects:
            self._p.removeBody(self.objects)
        if self.self_collision:
            self.objects = self._p.loadURDF(full_path,
                basePosition=self.basePosition,
                baseOrientation=self.baseOrientation,
                useFixedBase=self.fixed_base,
                flags=pybullet.URDF_USE_SELF_COLLISION)
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                self.objects)
        else:
            self.objects = self._p.loadURDF(full_path,
                basePosition=self.basePosition,
                baseOrientation=self.baseOrientation,
                useFixedBase=self.fixed_base)
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                self.objects)
        # self.objects will contain a body unique id, non-negative integer value. 
        self.episode += 1
        print("\nEpisode {}.\n".format(self.episode))
        self.robot_specific_reset(self._p)
        self._p.stepSimulation()
        ##### Add Pybullet wrapper here so that state can be calculated since there will be ROS msg #####
        self.rosWrapper = pyBulletRosWrapper(self.objects, self._p)

        # for task in self.rosWrapper.plugins: 
        #    task.execute()
        if rospy.get_param('~parallel_plugin_execution', False):
            self.rosWrapper.start_pybullet_ros_wrapper_parallel() # no good. Gets stuck within the loop
        else:
            for task in self.rosWrapper.plugins: 
                task.execute()
        self.publish_goal() # publish 'global goal' to move_base. Function in nav_omnirobot.py
        norm_state, state = self.calc_state()
        ### 8 Dec 2020, to test network not taking in contact data but will be penalised for contact
        # s = np.delete(s, 10)
        # s = np.delete(s, 9)
        # s = np.delete(s, 8)
        # s = np.delete(s, 7)
        # s = np.delete(s, 6)
        # s = np.delete(s, 5)
        ### See how well LiDAR alone works
        # self.potential = self.calc_potential()

        return norm_state, self.rosWrapper

    @staticmethod
    def calc_potential():
        return 0



class pyBulletRosWrapper(object):
  """ROS wrapper class for pybullet simulator"""
  # condensed
  def __init__(self, objects, _p):
    # self.pb = importlib.import_module('pybullet')
    self.pb = _p
    self.loop_rate = rospy.get_param('~loop_rate', 80.0) # rospy.Rate(rospy.get_param('~loop_rate', 80.0))
    # physicsClient = self.start_gui(gui=True)
    self.robot_loadedURDF = objects
    self.pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    rev_joint_index_name_dic, fixed_joint_index_name_dic, link_names_to_ids_dic = self.get_properties()
    
    # print("\n\nChecking the link names to ids: ")
    # print(link_names_to_ids_dic)

    # import plugins dynamically
    self.plugins = []
    dic = rospy.get_param('~plugins', {})
    if not dic:
      rospy.logwarn('No plugins found, forgot to set param ~plugins?')
    # return to normal shell color
    print('\033[0m')
    for key in dic:
      #rospy.loginfo('loading plugin: %s class from %s', dic[key], key)
      # create object of the imported file class
      obj = getattr(importlib.import_module(key), dic[key])(self.pb, self.robot_loadedURDF,
                     rev_joints=rev_joint_index_name_dic,
                    fixed_joints=fixed_joint_index_name_dic,
                     link_ids=link_names_to_ids_dic)
        # store objects in member variable for future use
      self.plugins.append(obj)
    # rospy.loginfo('Condensed pybullet ROS wrapper initialized.')

  def get_properties(self):
    """
    construct 3 dictionaries:
    - joint index to joint name x2 (1 for revolute, 1 for fixed joints)
    - link name to link index dictionary
    """
    rev_joint_index_name_dic = {}
    fixed_joint_index_name_dic = {}
    link_names_to_ids_dic = {}
    for joint_index in range(0, self.pb.getNumJoints(self.robot_loadedURDF)):
      info = self.pb.getJointInfo(self.robot_loadedURDF, joint_index)
      # rospy.loginfo(info)
      # build a dictionary of link names to ids
      link_names_to_ids_dic[info[12].decode('utf-8')] = joint_index
      # ensure we are dealing with a revolute joint
      if info[2] == self.pb.JOINT_REVOLUTE:
        # insert key, value in dictionary (joint index, joint name)
        rev_joint_index_name_dic[joint_index] = info[1].decode('utf-8') # info[1] refers to joint name
      elif info[2] == self.pb.JOINT_FIXED:
        # insert key, value in dictionary (joint index, joint name)
        fixed_joint_index_name_dic[joint_index] = info[1].decode('utf-8') # info[1] refers to joint name
    return rev_joint_index_name_dic, fixed_joint_index_name_dic, link_names_to_ids_dic

  def start_gui(self, gui=True):
    """start physics engine (client) with or without gui"""
    if(gui):
      # start simulation with gui
      rospy.loginfo('Running pybullet with gui')
      rospy.loginfo('-------------------------')
      return self.pb.connect(self.pb.GUI)
    else:
      # start simulation without gui (non-graphical version)
      rospy.loginfo('Running pybullet without gui')
      # hide console output from pybullet
      rospy.loginfo('-------------------------')
      return self.pb.connect(self.pb.DIRECT)

  def pause_simulation_function(self):
    # return self.pause_simulation
    return False

  def start_pybullet_ros_wrapper_sequential(self):
    """main simulation control cycle:
    1) check if position, velocity or effort commands are available, if so, forward to pybullet
    2) query joints state (current position, velocity and effort) and publish to ROS
    3) perform a step in pybullet simulation
    4) sleep to control the frequency of the node
    """
    rate = rospy.Rate(self.loop_rate)
    while not rospy.is_shutdown():
      # if not self.pause_simulation:
        # run x plugins
      self.pb.stepSimulation()
      for task in self.plugins:
        task.execute()
        # perform all the actions in a single forward dynamics simulation step such
        # as collision detection, constraint solving and integration
        # self.pb.stepSimulation()
      # self.loop_rate.sleep()
      rate.sleep()
    # rospy.logwarn('killing node now...')
    # if node is killed, disconnect
    # if self.connected_to_physics_server:
    #  self.pybullet.disconnect()

  def start_pybullet_ros_wrapper(self):
    if rospy.get_param('~parallel_plugin_execution', True):
      self.start_pybullet_ros_wrapper_parallel()
    else:
      self.start_pybullet_ros_wrapper_sequential()

  def start_pybullet_ros_wrapper_parallel(self):
    """
    Execute plugins in parallel, however watch their execution time and warn if exceeds the deadline (loop rate)
    """
    # create object of our parallel execution manager
    exec_manager_obj = FuncExecManager(self.plugins, rospy.is_shutdown, self.pb.stepSimulation, self.pause_simulation_function,
                                       log_info=rospy.loginfo, log_warn=rospy.logwarn, log_debug=rospy.logdebug)
    # start parallel execution of all "execute" class methods in a synchronous way
    exec_manager_obj.start_synchronous_execution(loop_rate=self.loop_rate)
    # ctrl + c was pressed, exit
    '''rospy.logwarn('killing node now...')
    # if node is killed, disconnect
    if self.connected_to_physics_server:
      self.pb.disconnect()'''

class SDFBasedRobot(XmlBasedRobot):
    """
    Base class for SDF robots in a Scene.
    """

    def __init__(self, model_sdf, robot_name, action_dim, obs_dim, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], fixed_base=False, self_collision=False):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)

        self.model_sdf = model_sdf
        self.fixed_base = fixed_base

    def reset(self, bullet_client):
        self._p = bullet_client

        self.ordered_joints = []

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,  # TODO: Not sure if this works, try it with kuka
            self._p.loadSDF(os.path.join("models_robot", self.model_sdf)))

        self.robot_specific_reset(self._p)

        norm_state, s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()

        return norm_state

    @staticmethod
    def calc_potential():
        return 0


class PoseHelper:  # dummy class to comply to original interface
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()


class BodyPart:
    def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.bodyPartIndex = bodyPartIndex
        self.initialPosition = self.current_position()
        self.initialOrientation = self.current_orientation()
        self.bp_pose = PoseHelper(self)

    def state_fields_of_pose_of(self, body_id, link_id=-1):  # a method you will most probably need a lot to get pose and orientation
        if link_id == -1:
            (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
        return np.array([x, y, z, a, b, c, d])

    def get_pose(self):
        return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

    def speed(self):
        if self.bodyPartIndex == -1:
            (vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            (x, y, z), (a, b, c, d), _,_,_,_, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1)
        return np.array([vx, vy, vz])

    def current_position(self):
        return self.get_pose()[:3]

    def current_orientation(self):
        return self.get_pose()[3:]

    def get_position(self):
        return self.current_position()

    def get_orientation(self):
        return self.current_orientation()

    def get_velocity(self):
        return self._p.getBaseVelocity(self.bodies[self.bodyIndex])

    def reset_position(self, position):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, self.get_orientation())

    def reset_orientation(self, orientation):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.get_position(), orientation)

    def reset_velocity(self, linearVelocity=[0,0,0], angularVelocity =[0,0,0]):
        self._p.resetBaseVelocity(self.bodies[self.bodyIndex], linearVelocity, angularVelocity)

    def reset_pose(self, position, orientation):
        self._p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

    def pose(self):
        return self.bp_pose

    def contact_list(self):
        return self._p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)


class Joint:
    JOINT_REVOLUTE_TYPE = 0
    JOINT_PLANAR_TYPE = 1
    JOINT_PRISMATIC_TYPE = 2
    JOINT_SPHERICAL_TYPE = 3
    JOINT_FIXED_TYPE = 4

    def __init__(self, bullet_client, joint_name, bodies, bodyIndex, jointIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex
        self.joint_name = joint_name

        joint_info = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
        self.jointType = joint_info[2]
        self.lowerLimit = joint_info[8]
        self.upperLimit = joint_info[9]
        self.jointHasLimits = self.lowerLimit < self.upperLimit
        self.jointMaxVelocity = joint_info[11]
        self.power_coeff = 0

    def set_state(self, x, vx):
        self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

    def current_position(self):  # just some synonym method
        return self.get_state()

    def current_relative_position(self):
        pos, vel = self.get_state()
        if self.jointHasLimits:
            pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
            pos = 2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit)

        if self.jointMaxVelocity > 0:
            vel /= self.jointMaxVelocity
        elif self.jointType == 0:  # JOINT_REVOLUTE_TYPE
            vel *= 0.1
        else:
            vel *= 0.5
        return (
            pos,
            vel
        )

    def get_state(self):
        x, vx,_,_ = self._p.getJointState(self.bodies[self.bodyIndex], self.jointIndex)
        return x, vx

    def get_position(self):
        x, _ = self.get_state()
        return x

    def get_orientation(self):
        _,r = self.get_state()
        return r

    def get_velocity(self):
        _, vx = self.get_state()
        return vx

    def set_position(self, position):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex], self.jointIndex, pybullet.POSITION_CONTROL, targetPosition=position)

    def set_velocity(self, velocity):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex], self.jointIndex, pybullet.VELOCITY_CONTROL, targetVelocity=velocity)

    def set_motor_torque(self, torque):  # just some synonym method
        self.set_torque(torque)

    def set_torque(self, torque):
        self._p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex], jointIndex=self.jointIndex, controlMode=pybullet.TORQUE_CONTROL, force=torque)  # positionGain=0.1, velocityGain=0.1)

    def reset_current_position(self, position, velocity):  # just some synonym method
        self.reset_position(position, velocity)

    def reset_position(self, position, velocity):
        self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, targetValue=position, targetVelocity=velocity)
        self.disable_motor()

    def disable_motor(self):
        self._p.setJointMotorControl2(self.bodies[self.bodyIndex], self.jointIndex, controlMode=pybullet.POSITION_CONTROL, targetPosition=0, targetVelocity=0, positionGain=0.1, velocityGain=0.1, force=0)
