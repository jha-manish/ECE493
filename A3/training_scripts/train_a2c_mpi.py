
#!/usr/bin/env python3

import os
import gym
from slimevolleygym import SurvivalRewardEnv

from mpi4py import MPI
import slimevolleygym
from stable_baselines.a2c import a2c
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import bench, logger
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_FREQ = 250000
EVAL_EPISODES = 1000
LOGDIR = "a2c_" # moved to zoo afterwards.

def make_env(seed):
    env = gym.make("SlimeVolley-v0")
    env.seed(seed)
    return env

def train():

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logger.configure(folder=LOGDIR)
    else:
        logger.configure(format_strs=[])

    workerseed = SEED + 10000*rank
    set_global_seeds(workerseed)
    env = make_env(workerseed)

    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    #model = a2c.A2C(MlpPolicy, env, n_steps=100000, gamma=0.99, learning_rate=0.01, lr_schedule='linear', verbose=2)
    model = a2c.A2C(MlpPolicy, env, n_steps=5, gamma=0.99, learning_rate=0.0007, lr_schedule='constant', max_grad_norm=0.5, vf_coef=0.25, ent_coef=0.01, alpha=0.99, epsilon= 1e-05, verbose=2)
    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=10, render=True)
    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)
    env.close()

    del env
    if rank == 0:
        model.save(os.path.join(LOGDIR, "final_model"))


if __name__ == '__main__':
    train()

