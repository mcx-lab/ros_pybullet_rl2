# ros_pybullet_rl2

An integration between [ROS](www.ros.org) and [PyBullet](https://github.com/bulletphysics/bullet3) and [OpenAI Gym Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for doing reinforcement and imitation learning on a ROS robot in a Pybullet simulation. 

<img src="https://github.com/mcx-lab/ros_pybullet_rl2/blob/master/common/sample_run.png" alt="show" />

# Project status

This project is an upgrade of the original ros_pybullet_rl package at (https://github.com/yungchuenng/ros_pybullet_rl_), in line with Stable Baselines 2's upgrade from https://github.com/hill-a/stable-baselines to Stable Baselines 3. It currently presents the following features:

- Omnirobot urdf model (omnirobot_v3). 

- The environment and robot scripts for RL training of an Omnirobot for dynamic obstacle avoidance under the Pybullet Gymperium framework. 

- ROS sensors plugin into Pybullet robot model: Robot base odometry, laser scanner, camera, force torque sensors. 

- ira_laser_tools (citing: Augusto and Pietro at http://wiki.ros.org/ira_laser_tools)

- Adjustable hyperparameters yml files for using the selected RL algorithms as found in src/ros_pybullet_rl2/hyperparams. 

- Can be improved:

        - Publishing frequency of the /scan and /d_image topics. 

- Limitation:

        - Training can only be done in 1 environment (thread) at a time because of the nature of ROS.

- Future Work:

        - DAgger training is still unready as we are unsure that how we should input the n_timesteps variable in *ros_pybullet_rl2/config/omnirobot_training_params.yaml*. As of now, DAgger training will not complete with all the n_timesteps value we have tried. We found out that DAgger training is outputing expert data instead of trained model and we can not validate the performance at the end of this FYP. We can put more focus on studying how to conduct a proper DAgger training with the correct variable in the future.

        - Video of Teacher Student networks that can be implemented with this framework in the future: https://youtu.be/TWx7LmiOyzA?t=1896

The main Reinforcement Learning training code is [here](https://github.com/mcx-lab/ros_pybullet_rl2/blob/master/src/ros_pybullet_rl2/ros_pybullet_rl2.py)
The main Imitation Learning code is [here](https://github.com/mcx-lab/ros_pybullet_rl2/tree/gail/src/ros_pybullet_rl2)


## Installation of ros_pybullet_rl2

The following instructions have been tested under **Ubuntu 16.04** with **ROS Kinetic distribution** and **Ubuntu 18.04** with **ROS Melodic distribution**. 

Clone this repository into your ROS workspace directory, your_workspace_ws/src. Rename this package into "ros_pybullet_rl2". 

Use a **Python 3.7 environment** for the setup. 

Install the following dependencies in your setup/environment: 

        sudo apt install ros-kinetic-cv-bridge
        sudo apt install ros-kinetic-navigation
        sudo apt install ros-kinetic-eband-local-planner
        sudo apt install ros-kinetic-hector-slam
        sudo apt install make gcc libgtk-3-dev libwebkitgtk-dev libwebkitgtk-3.0-dev freeglut3 freeglut3-dev python-gst-1.0 python3-gst-1.0 libglib2.0-dev ubuntu-restricted-extras libgstreamer-plugins-base1.0-dev
        pip install stable-baselines3[extra]
        pip install -r requirements.txt


**Note**: If you are using **Ubuntu 18.04** and **ROS Melodic distribution**, the installation for wxpython, cv-bridge, ROS Navigation Stack and Hector SLAM differs:

        sudo apt install ros-melodic-opencv-apps
        sudo apt install ros-melodic-navigation
        sudo apt install ros-melodic-eband-local-planner
        sudo apt install ros-melodic-hector-slam

Check that **ros-melodic-cv-bridge** is installed and **rospkg** installed in python3.6 environment.

To install Pybullet-Gym, clone repository into any desired location and install locally:

        git clone https://github.com/benelot/pybullet-gym.git
        cd pybullet-gym
        pip install -e .


## Installation of imitation

```
pip install imitation
```

```
git clone http://github.com/HumanCompatibleAI/imitation
cd imitation
pip install -e .
```

Then do the following:

- Move the directory, ros_pybullet_rl2/common/additions_into_pybullet-gym/yc, into pybullet-gym/pybulletgym/envs. 

- Move the file, ros_pybullet_rl2/common/additions_into_pybullet-gym/__init__.py, into pybullet-gym/pybulletgym/envs. 

- Move the directory, ros_pybullet_rl2/common/other_packages/ira_laser_tools, into your_workspace_ws/src.

- Move the directory, ros_pybullet_rl2/common/other_packages/omnirobot_v3, into your_workspace_ws/src. 

- Move the directory, ros_pybullet_rl2/common/other_packages/semantic_layers, into your_workspace_ws/src. 

- Move the directory, ros_pybullet_rl2/common/additions_into_pybullet-gym/assets, into pybullet-gym/pybulletgym/envs. Copy and merge. 

- Move the file, ros_pybullet_rl2/common/additions_into_imitation/rl.py, into imitation/src/imitation/scripts/common. Replace.

- Move the file, ros_pybullet_rl2/common/additions_into_cartographer/configuration_files/revo_lds.lua, into cartographer_workspace_ws/src/cartographer_ros/cartographer_ros/configuration_files. Replace.

- Move the file, ros_pybullet_rl2/common/additions_into_cartographer/launch/demo_revo_lds.launch, into cartographer_workspace_ws/src/cartographer_ros/cartographer_ros/launch. Replace.

Navigate to your_workspace_ws and build the package:
 
        catkin_make

To make sure .../your_workspace_ws/src shows in the **$ROS_PACKAGE_PATH** environment variable, in the same directory, run: 

        source devel/setup.bash

To ensure safe loading of configuration file parameters to avoid potential error crashing in between training process, replace *.../roslaunch/loader.py*with ros_pybullet_rl2/common/other_packages/roslaunch/loader.py (replace the **kinetic** with **melodic** accordingly depending on version) at:
```
/opt/ros/kinetic/lib/python2.7/dist-packages/roslaunch/loader.py
```

## Run the RL, Record of expert data for Imitation Learning, GAIL, AIRL, and DAgger training

- To edit the configuration files for training:

Navigate to *ros_pybullet_rl2/config*, and edit *omnirobot_training_params.yaml* **(Note NavOmnibase-v1 can only be run with ROS Kinetic)**. 
i.e. make sure to change the **tensorboard_log** input to your desired location to record the log of the reinforcemeant learning training process. 

- Before starting GAIL training, change the **expert_data** input to your desired final.pkl location
- For GAIL training, **n_timesteps** input need to be !!int instead of !!float while RL training can be both.
- **rollout_save_n_episodes** input change the size of the expert data. For example, **rollout_save_n_episodes** = 10 will save expert data for every 10 episodes.  

<img src="https://github.com/mcx-lab/ros_pybullet_rl2/blob/gail/common/training_config.png" alt="show" />

Run the RL training:

        roslaunch ros_pybullet_rl2 nav_train_rl.launch

Record of expert data:

        roslaunch ros_pybullet_rl2 nav_human_demo.launch

Run the GAIL training:

        roslaunch ros_pybullet_rl2 nav_train_gail.launch

Run the AIRL training:

        roslaunch ros_pybullet_rl2 nav_train_airl.launch

Run the DAgger training:

        roslaunch ros_pybullet_rl2 nav_train_dagger.launch

- To monitor the training of your robot at any instance in time (Note this is not real-time), navigate to *src/ros_pybullet_rl2* and run:

RL training: 

        python check_training_progress.py --model-path logs/ppo/NavOmnibase-v1_1

GAIL training: 

        python check_training_progress.py --model-path logs/ppo/NavOmnibase-v1_1/NavOmnibase-v1/checkpoints/monitor

*Change the model path according to the path to your model directory or log path. The trained model is saved in src/ros_pybullet_rl2/logs/algorithm_name/env_id by default.*

- To visualise the robot agent in **rviz**:

        rosrun rviz rviz

This is possible, but it may cause the computer to freeze if there is insufficient memory space or the CPU usage becomes fully clocked. This is able to visualise the robot model and the /scan, /rgb_image, /odom topics. Select the relevant topics for visualisation. 

- To load a previously trained agent for further training:

Navigate to *ros_pybullet_rl2/config*, and edit *omnirobot_training_params.yaml*. Change the **trained_agent** parameter to the relative path (from *ros_pybullet_rl2.py*) of your trained agent directory, i.e. *logs/ppo2/Omnibase-v1_1*. 

- The default hyperparameters used for training may also be override by specifying in the *omnirobot_training_params.yaml*. To change the neural network architecture i.e. from 2 layers of 64 nodes to 4 layers of 64 nodes, edit **net_arch=[dict(pi=[64, 64], vf=[64, 64])]** to **net_arch=[dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64])]**: 

```
hyperparams:                {'policy': 'MlpPolicy',
                              'policy_kwargs': "dict(log_std_init=-2,
                               ortho_init=False,
                               activation_fn=nn.ReLU,
                               net_arch=[dict(pi=[64, 64], vf=[64, 64])]
                               )"
}
```

- To change the input data mode to the neural network, edit the **obs_input_type** parameter (take note that the RL environment has to support the use of this parameter). The choices are ['default','array', 'multi_input']. 'array' ('defaullt') is for use with MlpPolicy, 'multi_input' is for use with MultiInputPolicy. 
```
obs_input_type:             array
```


- To choose a simulation environment for training:

Navigate to *config/omnirobot_training_params.yaml*. Edit the **select_env** parameter according to the number ID of the preset environment you would like to train the robot in. 

0. Stadium maze environment
1. Simple straight line
2. Curvy corridor
3. Twisting corridor
4. Straight line with dynamic obstacles 
5. T-junction with unseen static obstacles
6. T-junction with dynamic obstacles
7. Hall, goals in circular formation
8. Static environment with deadends
9. Dynamic obstacles environment without static obstacles
10. Lab
11. Lab (Dynamic)
12. Doorway
13. Large continuos environment

As of now, the average rate at which the relevant sensors and output command are published for use in reinforcement learning are recorded as follows:

        - /depth_image_processed: 0.420 Hz 
        - /contact_sensor: 42 Hz 
        - /scan: 5.3 Hz 
        - /odom: 42 Hz 
        - /cmd_vel: 55 Hz

To set the number of timesteps at which to do an evaluation of performance (not policy evaluation) (i.e. reward mean) of the agent in training:  
        - n_steps in ros_pybullet_rl2/src/ros_pybullet_rl2/hyperparams/{algorithm_name}.yml

To set the number of timesteps at which to do a policy evaluation for the training agent: 
        - eval_freq in config/omnirobot_training_params.yaml

- To fix the set of goals to be achieved by the robot agent in each environment:

Navigate to config/env_pybullet_params.yaml and under each environment, the **goal_set** variable consisting of a list of tuples may be edited to change the set of goals to be reached for the learning agent. If goal_set is set to '[]', random goals will be generated in the environment. 

## Run the validation of trained models

First, place the trained model, i.e. Omnibase-v1_1, found in src/ros_pybullet_rl2/logs/algorithm_name directory (by default), into src/ros_pybullet_rl2/trained_agents/allgorithm_name directory, for example:
```
sudo cp -r /root/workspace_ws/src/ros_pybullet_rl2/src/ros_pybullet_rl2/logs/ppo/NavOmnibase-v1_1 /root/workspace_ws/src/ros_pybullet_rl2/src/ros_pybullet_rl2/trained_agents/ppo
```

- To edit the configuration files for validation:

Navigate to ros_pybullet_rl2/config, and edit omnirobot_validation_params.yaml. Note that the **load_best** parameter when set to True indicates that the best_model.zip in the trained model directory will be loaded for validation, else the latest trained model.zip will be loaded instead. 
Set the number of timesteps to run the validation program by changing the **n_timesteps** parameter as required. 

Run the validation program:

        roslaunch ros_pybullet_rl2 nav_validate.launch

- To choose an environment for validation:

Navigate to config/omnirobot_validation_params.yaml. Edit **select_env** parameter according to the number ID of the preset environment you would like to train the robot in.

0. Stadium maze environment
1. Simple straight line
2. Curvy corridor
3. Twisting corridor
4. Straight line with dynamic obstacles 
5. T-junction with unseen static obstacles
6. T-junction with dynamic obstacles
7. Hall, goals in circular formation
8. Static environment with deadends
9. Dynamic obstacles environment without static obstacles
10. Lab
11. Lab (Dynamic)
12. Doorway
13. Large continuos environment

- To fix the set of goals to be achieved by the robot agent in each environment:

Navigate to config/env_pybullet_params.yaml and under each environment, the **validation_goal_set** variable consisting of a list of tuples may be edited to change the set of goals to be reached for the trained agent. If goal_set is set to '[]', random goals will be generated in the environment. 

## Visualise training process

- To visualise the training process using Tensorboard visualisation toolkit by TensorFlow: 

According to the **tensorboard_log** directory location, i.e. /home/ubuntu/your_directory_name, as set in "Test the RL training", navigate to the directory containing this directory and run: 

        tensorboard --logdir your_directory_name

Following the instructions as printed on the terminal, run the given link on a browser to start up the Tensorboard window, for instance, **http://my_computer_name:6006**.


## Pybullet robot agent sensors retrofitting

- To choose the relevant sensors for initialisation on the simulation robot model:

Navigate to config/omnirobot_pybullet_params.yaml. Here, the plugin sensors to fit onto the Pybullet robot model may be introduced into the **plugins** parameter. The relevant sensor information will have to set accordingly below, for LiDAR, RGBD Camera, force-torque sensors etc. The rate at which the Pybullet simulation environment runs may also be controlled via the parameter **loop_rate**. 

- To introduce a new sensor plugin to the robot model:

Navigate to src/ros_pybullet_rl2/plugins. The sensor plugins equipped onto the robot is found here. Refer to plugin_template.py for a template to create a custom plugin. 


## Standard Pybullet environment construction

The Pybullet simulation environment is created based on the createMultiBody function. The parameters of the generated bodies are written in config/env_pybullet_params.yaml. 

- To create new environments for simulation:

Navigate to env_pybullet_params.yaml, imitating the format used to construct the existing standard environments, existing environments may be edited and new environments may be created. 



