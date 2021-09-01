import os
from time import sleep
import sys
import pkg_resources
import importlib
import warnings

import rospy

import gym
import pybulletgym
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import yaml
import ros_pybullet_rl2.utils.import_envs  # pytype: disable=import-error
import numpy as np
import torch as th

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from geometry_msgs.msg import Twist, WrenchStamped

from ros_pybullet_rl2.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from ros_pybullet_rl2.utils.utils import StoreDict
from ros_pybullet_rl2.utils.exp_manager import ExperimentManager

# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer

class inputArguments():
     def __init__(self):
        self.env = rospy.get_param('~env')
        self.log_folder = rospy.get_param('~log_folder') # trained_agent
        self.algo = rospy.get_param('~algo')
        self.n_timesteps = rospy.get_param('~n_timesteps')
        self.num_threads = rospy.get_param('~num_threads')
        self.n_envs = rospy.get_param('~n_envs')
        self.exp_id = rospy.get_param('~exp_id')
        self.verbose = rospy.get_param('~verbose')
        self.render = rospy.get_param('~render')
        self.deterministic = rospy.get_param('~deterministic')
        self.stochastic = rospy.get_param('~stochastic')
        self.load_best = rospy.get_param('~load_best')
        self.load_checkpoint = rospy.get_param('~load_checkpoint')
        self.load_last_checkpoint = rospy.get_param('~load_last_checkpoint')
        self.norm_reward = rospy.get_param('~norm_reward')
        self.seed = rospy.get_param('~seed')
        self.reward_log = rospy.get_param('~reward_log')
        self.gym_packages = rospy.get_param('~gym_packages')
        self.env_kwargs = eval(rospy.get_param('~env_kwargs'))

def main():
    rospy.init_node('ros_pybullet_rl2_validation', anonymous=False, disable_signals=True)
    # vel_pub = rospy.Publisher('/omnibase/cmd_vel', Twist, queue_size=5)

    args = inputArguments()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    log_folder = args.log_folder

    ros_pybullet_rl_dir = rospy.get_param('~ros_pybullet_rl2_dir')
    log_path = "{}/{}/{}/".format(ros_pybullet_rl_dir, args.log_folder, args.algo)

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(log_path, env_id)
        print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} log_folder was not found".format(log_path)

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if args.load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, "rl_model_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {algo} on {env_id}, path: {log_path}")

        def step_count(checkpoint_path: str) -> int:
            # path follow the pattern "rl_model_*_steps.zip", we count from the back to ignore any other _ in the path
            return int(checkpoint_path.split("_")[-2])

        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
        found = True

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = False

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    args.reward_log = "{}/{}/{}/{}_{}/".format(ros_pybullet_rl_dir, args.reward_log, args.algo, env_id, args.exp_id)
    log_dir = args.reward_log if args.reward_log != '' else None

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    env = create_test_env(env_id, 
                          n_envs=args.n_envs, 
                          # is_atari=False,
                          stats_path=stats_path, 
                          seed=args.seed, 
                          log_dir=log_dir,
                          should_render=args.render,
                          hyperparams=hyperparams, 
                          env_kwargs=env_kwargs)

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    if args.render:
        env.render(mode='human')
    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    state = None

    calc_timestep = 0
    x_vel_list = []
    y_vel_list = []
    th_vel_list = []
    time_start = rospy.Time.now()

    try:
        for _ in range(args.n_timesteps): # from episode 9, the robot model disappears from Pybullet... 
            if calc_timestep:
                time_t1 = (rospy.Time.now() - time_t).to_sec()
                print(1/time_t1, " Hz")
                calc_timestep = 0
            if _ + 1 % 1000 == 0:
                print("Total timestep elapsed: ", _)
                calc_timestep = 1
                time_t = rospy.Time.now()

            action, state = model.predict(obs, state=state, deterministic=deterministic)

            x_vel_list.append(action[0][0])
            y_vel_list.append(action[0][1])
            th_vel_list.append(action[0][2])

            # Clip Action to avoid out of bound errors
            # if isinstance(env.action_space, gym.spaces.Box):
            #     action = np.clip(action, env.action_space.low, env.action_space.high)
            
            obs, reward, done, infos = env.step(action)

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
                    # if len(episode_rewards) == 8: # Halt validation after 8 episodes. 
                    #     break

                # Reset also when the goal is achieved when using HER
                if done and infos[0].get("is_success") is not None:
                    if args.verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0

                    # Alternatively, you can add a check to wait for the end of the episode
                    # if done:
                    # obs = env.reset() # Do not have to repeat reset here (for implementation together with Pybullet). 

    except KeyboardInterrupt:
        pass

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

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

