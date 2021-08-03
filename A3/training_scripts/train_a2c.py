#!/usr/bin/env python3

import os
import gym
from slimevolleygym import SurvivalRewardEnv

import slimevolleygym
from stable_baselines.a2c import a2c
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_FREQ = 250000
EVAL_EPISODES = 1000
LOGDIR = "a2c" # moved to zoo afterwards.

logger.configure(folder=LOGDIR)

env = gym.make("SlimeVolley-v0")
env.seed(SEED)

model = a2c.A2C(MlpPolicy, env, verbose=2)

eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES, render=True)

model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

env.close()
