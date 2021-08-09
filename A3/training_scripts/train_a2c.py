#!/usr/bin/env python3

import os
import gym
from slimevolleygym import SurvivalRewardEnv

from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import subproc_vec_env
import slimevolleygym
from stable_baselines.a2c import a2c
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger, bench
from stable_baselines.common.callbacks import EvalCallback
import numpy as np
# import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import MlpPolicy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_FREQ = 250000
EVAL_EPISODES = 1000
LOGDIR = "a2c_vec" # moved to zoo afterwards.
env_id = "SlimeVolleyNoFrameskip-v0"

logger.configure(folder=LOGDIR)

env = gym.make(env_id)
env = bench.Monitor(env, LOGDIR)
env.seed(SEED)

# env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(0)))

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=LOGDIR)
model = a2c.A2C(MlpPolicy, env, n_steps=20, learning_rate=0.001, lr_schedule='linear', verbose=2)

#model = a2c.A2C(MlpPolicy, env, n_steps=10000, learning_rate=0.001, lr_schedule='linear', verbose=2)
#eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES, render=True)

model.learn(total_timesteps=NUM_TIMESTEPS, callback=callback)

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

env.close()
