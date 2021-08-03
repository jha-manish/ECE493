#!/usr/bin/env python3

import os
import gym
from slimevolleygym import SurvivalRewardEnv

from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.vec_env import subproc_vec_env
import slimevolleygym
from stable_baselines.a2c import a2c
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger, bench
from stable_baselines.common.callbacks import EvalCallback

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_FREQ = 250000
EVAL_EPISODES = 100000
LOGDIR = "a2c" # moved to zoo afterwards.
env_id = "SlimeVolley-v0"

logger.configure(folder=LOGDIR)
num_cpu = 4  # Number of processes to use

# Create the vectorized environment
#env = subproc_vec_env.SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
#env = make_vec_env('SlimeVolley-v0', n_envs=4)

env = gym.make("SlimeVolley-v0")
env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(0)))
env.seed(SEED)

model = a2c.A2C(MlpPolicy, env, n_steps=10000, learning_rate=0.001, lr_schedule='linear', verbose=2)

eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=int(2e7), n_eval_episodes=EVAL_EPISODES)

model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

env.close()
