#!/usr/bin/env python3

import os
import time
import uuid
import difflib
import importlib
import signal
from threading import Thread
from time import sleep 

import rospy

import gym
import pybulletgym
import numpy as np
import seaborn
import torch as th

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy

# Register custom envs
import ros_pybullet_rl2.utils.import_envs

from ros_pybullet_rl2.utils.exp_manager import ExperimentManager
from ros_pybullet_rl2.utils.utils import ALGOS, StoreDict, get_latest_run_id

import pathlib
import pickle
import tempfile
import sacred

import stable_baselines3 as sb3

from imitation.algorithms import bc
from imitation.algorithms import dagger
from imitation.data import rollout
from imitation.util import logger, util
from imitation.policies import serialize
from imitation.scripts.common import common as common_config
from imitation.scripts.common import demonstrations, reward, rl, train
from imitation.scripts import train_adversarial

seaborn.set()

global env_kwargs
global hyperparams

class inputArguments():
    def __init__(self):
        self.env = rospy.get_param('~env')
        self.tensorboard_log = rospy.get_param('~tensorboard_log')
        self.trained_agent = rospy.get_param('~trained_agent')
        self.expert_data = rospy.get_param('~expert_data')
        self.algo = rospy.get_param('~algo')
        self.n_timesteps = rospy.get_param('~n_timesteps')
        self.save_timesteps = rospy.get_param('~save_timesteps')
        self.rollout_save_n_timesteps = rospy.get_param('~rollout_save_n_timesteps')
        self.rollout_save_n_episodes = rospy.get_param('~rollout_save_n_episodes')
        self.num_threads = rospy.get_param('~num_threads')
        self.log_interval = rospy.get_param('~log_interval')
        self.eval_freq = rospy.get_param('~eval_freq')
        self.eval_episodes = rospy.get_param('~eval_episodes')
        self.n_eval_envs = rospy.get_param('~n_eval_envs')
        self.save_freq = rospy.get_param('~save_freq')
        self.save_replay_buffer = rospy.get_param('~save_replay_buffer')
        self.normalise_obs = rospy.get_param('~normalise_obs')
        self.log_folder = rospy.get_param('~log_folder')
        self.seed = rospy.get_param('~seed')
        self.vec_env = rospy.get_param('~vec_env')
        self.max_ram_usage = rospy.get_param('~max_ram_usage')

        self.n_trials = rospy.get_param('~n_trials')
        self.optimize_hyperparameters = rospy.get_param('~optimize_hyperparameters')
        self.optimization_log_path = rospy.get_param('~optimization_log_path')
        self.no_optim_plots = rospy.get_param('~no_optim_plots')
        self.n_jobs = rospy.get_param('~n_jobs')
        self.sampler = rospy.get_param('~sampler')
        self.pruner = rospy.get_param('~pruner')
        self.n_startup_trials = rospy.get_param('~n_startup_trials')
        self.n_evaluations = rospy.get_param('~n_evaluations')
        self.storage = rospy.get_param('~storage')
        self.study_name = rospy.get_param('~study_name')

        self.verbose = rospy.get_param('~verbose')
        self.gym_packages = rospy.get_param('~gym_packages')
        self.hyperparams = rospy.get_param('~hyperparams')
        self.uuid = rospy.get_param('~uuid')
        self.env_kwargs = eval(rospy.get_param('~env_kwargs'))
        self.truncate_last_trajectory = rospy.get_param('~truncate_last_trajectory')

        self.render = rospy.get_param('~render')
        self.select_env = rospy.get_param('~select_env')

def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    os.makedirs(save_path, exist_ok=True)
    th.save(trainer.reward_train, os.path.join(save_path, "reward_train.pt"))
    th.save(trainer.reward_test, os.path.join(save_path, "reward_test.pt"))
    # TODO(gleave): unify this with the saving logic in data_collect?
    # (Needs #43 to be merged before attempting.)
    serialize.save_stable_model(
        os.path.join(save_path, "gen_policy"),
        trainer.gen_algo,
        trainer.venv_norm_obs,
    )

def run():
    rospy.init_node('ros_pybullet_rl2_training', anonymous=False, disable_signals=True)

    args = inputArguments()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = '_{}'.format(uuid.uuid4()) if args.uuid else ''
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    set_random_seed(args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    exp_manager = ExperimentManager(
        args,
        args.algo,
        env_id,
        args.log_folder,
        args.tensorboard_log,
        args.n_timesteps,
        args.eval_freq,
        args.eval_episodes,
        args.save_freq,
        args.hyperparams,
        args.env_kwargs,
        args.trained_agent,
        args.optimize_hyperparameters,
        args.storage,
        args.study_name,
        args.n_trials,
        args.n_jobs,
        args.sampler,
        args.pruner,
        args.optimization_log_path,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        truncate_last_trajectory=args.truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
        n_eval_envs=args.n_eval_envs,
        no_optim_plots=args.no_optim_plots,
        max_ram_usage=args.max_ram_usage,
        save_timesteps=args.save_timesteps,
        rollout_save_n_timesteps=args.rollout_save_n_timesteps,
        rollout_save_n_episodes=args.rollout_save_n_episodes,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    # model = exp_manager.setup_experiment()

    ros_pybullet_rl2_dir = rospy.get_param('~ros_pybullet_rl2_dir')
    trained_agent_path = os.path.join(ros_pybullet_rl2_dir, args.trained_agent)
    expert_data_path = os.path.join(ros_pybullet_rl2_dir, args.expert_data)
    log_path = f"{ros_pybullet_rl2_dir}/{args.log_folder}/{args.algo}/"
    save_path = os.path.join(
        log_path, f"{env_id}_{get_latest_run_id(log_path, env_id)}{uuid_str}"
        )
    params_path = f"{save_path}/{env_id}"
    monitor_path = os.path.join(params_path, "checkpoints")

    custom_logger = logger.configure(os.path.join(args.tensorboard_log, "log"), ["tensorboard", "stdout", "log", "csv"])
    expert_trajs = demonstrations.load_expert_trajs(rollout_path=expert_data_path, n_expert_demos=None)

    venv = common_config.make_venv(env_name=env_id, num_vec=1, parallel=False, log_dir=monitor_path, max_episode_steps=1000, env_make_kwargs=None)
    # reward_net = reward.make_reward_net(venv, net_cls=None, net_kwargs=None)

    hyperparams, saved_hyperparams = exp_manager.read_hyperparameters()
    hyperparams, env_wrapper, callbacks = exp_manager._preprocess_hyperparams(hyperparams)

    _hyperparams = exp_manager._preprocess_action_noise(hyperparams, saved_hyperparams, venv)
    del _hyperparams["policy"]
    del _hyperparams["n_steps"]
    policy_kwargs = _hyperparams["policy_kwargs"]
    del _hyperparams["policy_kwargs"]
    print(_hyperparams)

    if args.trained_agent is not "":
        trained_agent = sb3.PPO.load(trained_agent_path)
    else:
        trained_agent = None

    gen_algo = rl.make_rl_algo(venv, sb3.PPO, batch_size=4096, rl_kwargs=_hyperparams, 
        train={'policy_cls':sb3.common.policies.ActorCriticPolicy, 'policy_kwargs':policy_kwargs, 
        'n_episodes_eval':args.eval_episodes, 'algorithm_specific': {}}, trained_agent = trained_agent)


    if args.tensorboard_log is not "":
        init_tensorboard = True
        init_tensorboard_graph = True

    # obs_dim = 32
    # if self.normalise:
    #     high = np.ones([obs_dim]) # * np.inf
    # else:
    #     high = np.ones([obs_dim]) * np.inf
    # observation_space = gym.spaces.Box(-high, high) # the range of input values

    # action_space = gym.spaces.Box(low=np.array([-1,-1,-2]), high=np.array([1,1,2]), dtype=np.float32)

    # Normal training
    if args.expert_data is not "":
        # exp_manager.learn(model)
        # exp_manager.save_trained_model(model)

        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=expert_trajs,
            batch_size=_hyperparams['batch_size'],
            optimizer_cls=th.optim.Adam,
            ent_weight = _hyperparams['ent_coef'], # ent_weight=1e-3,
            l2_weight=0.0,
            device="auto",
            custom_logger=custom_logger,
        )

        dagger_trainer = dagger.SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=os.path.join(monitor_path, "final"),
            expert_policy=gen_algo.policy,
            bc_trainer=bc_trainer,
            # demonstrations=expert_trajs,
            # batch_size=32,
            # optimizer_cls=th.optim.Adam,
            # ent_weight=1e-3,
            # l2_weight=0.0,
            # device="auto", 
            # # gen_algo=sb3.PPO(venv, verbose=1, **_hyperparams),
            # gen_algo=gen_algo,
            custom_logger=custom_logger,
            # log_dir=args.tensorboard_log,
            # init_tensorboard=init_tensorboard,
            # init_tensorboard_graph=init_tensorboard_graph,
            # allow_variable_horizon=True,
        )
        dagger_trainer.train(total_timesteps=args.n_timesteps, rollout_round_min_episodes=args.rollout_save_n_episodes, rollout_round_min_timesteps=args.rollout_save_n_timesteps, bc_train_kwargs={"n_epochs": _hyperparams['n_epochs']})

        dagger_trainer.save_policy(os.path.join(monitor_path, "model.zip")) # "final"))
    else:
        print("There is no expert data to run DAgger training, please check the expert_data parameter!")

    rospy.signal_shutdown('Training complete. Shutting down.\n________________________________')
