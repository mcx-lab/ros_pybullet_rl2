import os
from time import sleep
import sys
import argparse
import pkg_resources
import importlib
import warnings

import rospy

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import pybulletgym
import yaml
import ros_pybullet_rl.utils.import_envs  # pytype: disable=import-error
import numpy as np
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from geometry_msgs.msg import Twist, WrenchStamped

from ros_pybullet_rl.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model
from ros_pybullet_rl.utils.utils import StoreDict

# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer

class inputArguments():
    # def __init__(self, env, tb, i, algo, n, log_interval, eval_freq, eval_episodes, save_freq, f, seed, n_trials, optimize, n_jobs, samplers, pruner, verbose, gym_packages, params, uuid, env_kwargs):
    def __init__(self):
        self.env = rospy.get_param('~env')
        self.log_folder = rospy.get_param('~log_folder') # trained_agent
        self.algo = rospy.get_param('~algo')
        self.n_timesteps = rospy.get_param('~n_timesteps')
        self.n_envs = rospy.get_param('~n_envs')
        self.exp_id = rospy.get_param('~exp_id')
        self.verbose = rospy.get_param('~verbose')
        self.render = rospy.get_param('~render')
        self.deterministic = rospy.get_param('~deterministic')
        self.stochastic = rospy.get_param('~stochastic')
        self.load_best = rospy.get_param('~load_best')
        self.norm_reward = rospy.get_param('~norm_reward')
        self.seed = rospy.get_param('~seed')
        self.reward_log = rospy.get_param('~reward_log')
        self.gym_packages = rospy.get_param('~gym_packages')
        self.env_kwargs = eval(rospy.get_param('~env_kwargs'))

def main():
    rospy.init_node('pybullet_ros_rl_validation', anonymous=False, disable_signals=True)
    # vel_pub = rospy.Publisher('/omnibase/cmd_vel', Twist, queue_size=5)

    args = inputArguments()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    log_folder = args.log_folder

    ros_pybullet_rl_dir = rospy.get_param('~ros_pybullet_rl_dir')
    log_path = "{}/{}/{}/".format(ros_pybullet_rl_dir, args.log_folder, args.algo)

    if args.exp_id == 0:
        # args.exp_id = get_latest_run_id(os.path.join(log_folder, algo), env_id)
        args.exp_id = get_latest_run_id(log_path, env_id)
        print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        # log_path = os.path.join(log_folder, algo, '{}_{}'.format(env_id, args.exp_id))
        log_path = os.path.join(log_path, '{}_{}'.format(env_id, args.exp_id))
    # else:
        # log_path = os.path.join(log_folder, algo)

    assert os.path.isdir(log_path), "The {} log_folder was not found".format(log_path)

    model_path = find_saved_model(algo, log_path, env_id, load_best=args.load_best)

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        args.n_envs = 1

    set_global_seeds(args.seed)


    is_atari = False

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    args.reward_log = "{}/{}/{}/{}_{}/".format(ros_pybullet_rl_dir, args.reward_log, args.algo, env_id, args.exp_id)
    log_dir = args.reward_log if args.reward_log != '' else None

    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs

    env = create_test_env(env_id, n_envs=args.n_envs, is_atari=False,
                          stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                          should_render=args.render,
                          hyperparams=hyperparams, env_kwargs=env_kwargs)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    model = ALGOS[algo].load(model_path, env=load_env)

    if args.render:
        env.render(mode='human')
    obs = env.reset()

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    state = None

    # print("Checkpoint 1") # no issue
    calc_timestep = 0
    x_vel_list = []
    y_vel_list = []
    th_vel_list = []
    time_start = rospy.Time.now()
    # action_set = rospy.get_param()
    action_set = {'vel_x':[-0.2655, 0, 0.2655], 
                'vel_y':[-0.2655, 0, 0.2655], 
                'vel_th': [-0.2, 0, 0.2]}

    for _ in range(args.n_timesteps): # from episode 9, the robot model disappears from Pybullet... 
        if calc_timestep:
            time_t1 = (rospy.Time.now() - time_t).to_sec()
            print(1/time_t1, " Hz")
            calc_timestep = 0
        if _ + 1 % 1000 == 0:
            print("Total timestep elapsed: ", _)
            calc_timestep = 1
            time_t = rospy.Time.now()

        # print("before state: ", state)
        action, state = model.predict(obs, state=state, deterministic=deterministic)
        # print("after state: ", state)
        # print("action: ", action)
        
        # x_vel_list.append(action_set['vel_x'][action[0][0]])
        # y_vel_list.append(action_set['vel_y'][action[0][1]])
        # th_vel_list.append(action_set['vel_th'][action[0][2]])
        x_vel_list.append(action[0][0])
        y_vel_list.append(action[0][1])
        th_vel_list.append(action[0][2])


        '''vel_cmd = Twist()
        vel_cmd.linear.x = action_set['vel_x'][action[0][0]]
        vel_cmd.linear.y = action_set['vel_y'][action[0][1]]
        vel_cmd.angular.z = action_set['vel_th'][action[0][2]]'''

        # Random Agent
        # action = [env.action_space.sample()]
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        
        obs, reward, done, infos = env.step(action)
        
        # vel_pub.publish(vel_cmd)

        # if start:
        #    dist_to_goal = obs[0][0] # Don't use obs, obs is actually the weights
        #    start = 0

        episode_reward += reward[0]
        ep_len += 1

        if args.n_envs == 1:
            if done and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print("Episode Reward: {:.2f}".format(episode_reward))
                print("Episode Length", ep_len)

                try:
                    print("Time taken = ", (rospy.Time.now() - time_start).to_sec())
                    print("average x vel: ", sum(x_vel_list)/len(x_vel_list))
                    print("average y vel: ", sum(y_vel_list)/len(y_vel_list))
                    print("average th vel: ", sum(th_vel_list)/len(th_vel_list))
                    # print("position error: ", obs[0][0]/dist_to_goal*100, "%")
                    # print("Total reward in run: ", sum(reward_list))
                except:
                    print("Goal initiated within 0.30 m, or not initiated correctly.")
                x_vel_list = []
                y_vel_list = []
                th_vel_list = []
                time_start = rospy.Time.now()

                state = None
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)
                episode_reward = 0.0
                ep_len = 0
                if len(episode_rewards) == 8: # Halt validation after 8 episodes. 
                    break

            # Reset also when the goal is achieved when using HER
            if done or infos[0].get('is_success', False):
                if args.algo == 'her' and args.verbose > 1:
                    print("Success?", infos[0].get('is_success', False))
                # Alternatively, you can add a check to wait for the end of the episode
                # if done:
                # obs = env.reset() # Do not have to repeat reset here (for implementation together with Pybullet). 
                if args.algo == 'her':
                    successes.append(infos[0].get('is_success', False))
                    episode_reward, ep_len = 0.0, 0

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))

    if args.verbose > 0 and len(episode_lengths) > 0:
        print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))

    # Workaround for https://github.com/openai/gym/issues/893
    if args.render:
        if args.n_envs == 1 and 'Bullet' not in env_id and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            # SubprocVecEnv
            env.close()

    print("\nValidation complete.\n")

# if __name__ == '__main__':
#     main()
