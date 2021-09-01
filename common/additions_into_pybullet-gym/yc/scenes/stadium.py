import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)

from .scene_bases import Scene
import pybullet
import pybullet_data
import rospy

class StadiumScene(Scene):
	multiplayer = False
	zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the stadium
	stadium_halflen   = 105*0.25	# FOOBALL_FIELD_HALFLEN
	stadium_halfwidth = 50*0.25	 # FOOBALL_FIELD_HALFWID
	stadiumLoaded = 0

	def episode_restart(self, bullet_client):
		self._p = bullet_client
		Scene.episode_restart(self, bullet_client)
		if self.stadiumLoaded == 0:
			self.stadiumLoaded = 1

			#filename = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "scenes", "stadium", "plane_stadium.sdf")
			#self.ground_plane_mjcf=self._p.loadSDF(filename)
			filename = os.path.join(pybullet_data.getDataPath(),"stadium_no_collision.sdf")
			self.ground_plane_mjcf = self._p.loadSDF(filename)
			
			# Initialise list_attributes to query environmental bodies
			self.col_stc_bodies = []
			self.stc_bodies = []
			self.soft_bodies = []
			self.col_dyn_bodies = []
			self.dyn_bodies = []

			env_label = rospy.get_param('~select_env')
			self.env_obj = rospy.get_param('~env_obj{}/env'.format(str(env_label)))

			if rospy.get_param('~generate_env'):
				# load static bodies
				# path_to_obj = rospy.get_param('~env_obj_path')
				for obj in self.env_obj['static']:
					if self.env_obj['static'][obj]['shapeType'] == 'GEOM_BOX':
						cuid = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents = self.env_obj['static'][obj]['halfExtents'])
						self.col_stc_bodies.append(cuid)
						self.stc_bodies.append(self._p.createMultiBody(self.env_obj['static'][obj]['baseMass'], cuid, basePosition=self.env_obj['static'][obj]['basePosition']))
					if self.env_obj['static'][obj]['shapeType'] == 'GEOM_CYLINDER':
						cuid = self._p.createCollisionShape(self._p.GEOM_CYLINDER, radius = self.env_obj['static'][obj]['radius'], height = self.env_obj['static'][obj]['height'])
						self.col_stc_bodies.append(cuid)
						self.stc_bodies.append(self._p.createMultiBody(self.env_obj['static'][obj]['baseMass'], cuid, basePosition=self.env_obj['static'][obj]['basePosition']))
					if self.env_obj['static'][obj]['shapeType'] == 'GEOM_CAPSULE':
						cuid = self._p.createCollisionShape(self._p.GEOM_CAPSULE, radius = self.env_obj['static'][obj]['radius'], height = self.env_obj['static'][obj]['height'])
						self.col_stc_bodies.append(cuid)
						self.stc_bodies.append(self._p.createMultiBody(self.env_obj['static'][obj]['baseMass'], cuid, basePosition=self.env_obj['static'][obj]['basePosition']))
					# Note, GEOM_MESH creation will cause createCollisionShape to fail. 
					# if self.env_obj['static'][obj]['shapeType'] == 'GEOM_MESH':
					# 	cuid = self._p.createCollisionShape(self._p.GEOM_MESH, fileName = os.path.join(path_to_obj, self.env_obj['static'][obj]['fileName'])) #, meshScale = self.env_obj['static'][obj]['meshScale'])
					# 	self._p.createMultiBody(self.env_obj['static'][obj]['baseMass'], cuid, basePosition=self.env_obj['static'][obj]['basePosition'])
				# load dynamic bodies
				try:
					for obj in sorted(self.env_obj['dynamic'].keys()):
						if self.env_obj['dynamic'][obj]['shapeType'] == 'GEOM_CYLINDER':
							cuid = self._p.createCollisionShape(self._p.GEOM_CYLINDER, radius = self.env_obj['dynamic'][obj]['radius'], height = self.env_obj['dynamic'][obj]['height'])
							self.col_dyn_bodies.append(cuid)
							self.dyn_bodies.append(self._p.createMultiBody(self.env_obj['dynamic'][obj]['baseMass'], cuid, basePosition=self.env_obj['dynamic'][obj]['basePosition']))
					self.dyn_bodies_action_angle = rospy.get_param('~env_obj{}/action_angle_set'.format(str(env_label)))
				except:
					print("No dynamic bodies found.")
				#	loaded_sdf = self._p.loadSDF(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "scenes", "stadium", "%s.sdf" %model))
			
			elif rospy.get_param('~load_env'):
				urdf_directory = os.path.join(rospy.get_param("~main_dir"), "common", "urdf")
				# load static bodies
				for obj in self.env_obj['static']:
					self.stc_bodies.append(p.loadURDF(urdf_directory + '/' + self.env_obj['static'][obj]['filename'],
											self.env_obj['static'][obj]['basePosition'],
											self.env_obj['static'][obj]['Orientation'],
											globalScaling = self.env_obj['static'][obj]['globalScaling'],
											useMaximalCoordinates = True))
				# load soft bodies
				try:
					for obj in self.env_obj['static_soft']:
						self.soft_bodies.append(p.loadSoftBody(urdf_directory + '/' + self.env_obj['soft'][obj]['filename'],
									basePosition = self.env_obj['soft'][obj]['basePosition'],
									baseOrientation = self.env_obj['soft'][obj]['baseOrientation'],
									scale = self.env_obj['soft'][obj]['scale'],
									mass = self.env_obj['soft'][obj]['mass'],
									useNeoHookean = 0,useBendingSprings=1,useMassSpring=1,springDampingAllDirections = 1,
									springElasticStiffness=self.env_obj['soft'][obj]['springElasticStiffness'],
									springDampingStiffness=self.env_obj['soft'][obj]['springDampingStiffness'],
									collisionMargin = self.env_obj['soft'][obj]['collisionMargin'],
									frictionCoeff = self.env_obj['soft'][obj]['frictionCoeff'],
									useSelfCollision = 1, useFaceContact=1))
					# change colour
					# apply soft anchor
				except:
					print("No soft bodies found.")
				# load dynamic bodies
				try:
					for obj in sorted(self.env_obj['dynamic'].keys()):
						# to be expanded when involving dynamic bodies
						pass
				except:
					print("No dynamic bodies found.")

			for i in self.ground_plane_mjcf:
				self._p.changeDynamics(i,-1,lateralFriction=0.8, restitution=0.5)
				self._p.changeVisualShape(i,-1,rgbaColor=[1,1,1,1]) # rgbaColor=[1,1,1,0.8]
				self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,1)
