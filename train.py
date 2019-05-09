import random
import numpy as np
from collections import namedtuple

import gym
from gym import wrappers
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

import torch
import torch.optim as optim

from deepq.learn import mario_learning
from deepq.model import DQN

from common.atari_wrapper import wrap_mario
from common.schedule import LinearSchedule

SEED = 12345
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS    = 50000
LEARNING_FREQ      = 4 
FRAME_HISTORY_LEN  = 4
TARGET_UPDATE_FREQ = 5000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

def main(env):
    
    num_iterations = float(40000000) / 4.0

    exploration_schedule = LinearSchedule(1000000, 0.1)

    OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    mario_learning(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer,
        exploration=exploration_schedule,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

if __name__ == '__main__':
    
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = wrap_mario(env)

    output_dir = 'video/baseline'
    env = wrappers.Monitor(env, output_dir, force=True, video_callable=lambda count: count % 10 == 0)

    main(env)


