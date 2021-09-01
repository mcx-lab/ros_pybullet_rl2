from pybulletgym.envs.yc.robots.robot_bases2 import XmlBasedRobot
import numpy as np

# the purposes of this OmniBase class is so that other similar implementations of omnibase can share the same attributes.
class OmniBase(XmlBasedRobot):
    def __init__(self, goal_x, goal_y, start_pos_x):
        # self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = start_pos_x, 0, 0
        self.goal_x = goal_x # 1e3  # kilometer away
        self.goal_y = goal_y
        self.body_xyz = [0, 0, 0]

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        for j in self.ordered_joints: # consider changing these numbers to see how they affect reset
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0) 

        # self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        try:
            self.scene.actor_introduce(self)
        except AttributeError:
            pass
        self.initial_z = None

