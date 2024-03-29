import os
import sys
import random
import numpy as np
from itertools import count

import gym

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import pickle

from .replay_buffer import ReplayBuffer

# detect GPU
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
            
#########            
# Mario #
#########

def mario_learning(
    env,
    q_func,
    model,
    optimizer_spec,
    exploration,
    replay_buffer_size=600000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    ):
    
    assert type(env.observation_space) == gym.spaces.Box
    #assert type(env.action_space)      == gym.spaces.MultiDiscrete
    assert type(env.action_space)      == gym.spaces.Discrete
    
    
    if len(env.observation_space.shape) == 1:
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c  
    
    #num_actions = env.action_space.shape
    num_actions = len(COMPLEX_MOVEMENT)
    
    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, act_hot, obs, last_obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            act_hot = torch.from_numpy(act_hot).type(dtype).unsqueeze(0)
            last_obs = torch.from_numpy(last_obs).type(dtype)
            if type(model).__name__ == "DQN":
                with torch.no_grad():
                    return model(Variable(obs)).data.max(1)[1].cpu()
            else:
                with torch.no_grad():
                    return model(Variable(act_hot), Variable(obs), Variable(last_obs)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])
        
    # to one hot
    def to_onehot(action, num_actions):
        action = action % num_actions
        if action == 0:
            # Move right while jumping
            action_onehot = np.zeros(num_actions, dtype=int)
            action_onehot[2] = 1
        else:
            action_onehot = np.zeros(num_actions, dtype=int)
            action_onehot[action] = 1
        return action_onehot


    # Initialize target q function and q function
    Q        = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)
    
    # Check & load pretrain model
    if os.path.isfile('mario_Q_params.pkl'):
        print('Load Q parameters ...')
        Q.load_state_dict(torch.load('mario_Q_params.pkl'))
        
    if os.path.isfile('mario_target_Q_params.pkl'):
        print('Load target Q parameters ...')
        target_Q.load_state_dict(torch.load('mario_target_Q_params.pkl'))
    
    
    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
    
    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, num_actions)
    
    ### RUN ENV
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000 #TODO change back
    
    num_trials = 0
    loss_arr = []
    reward_arr = []
    trial_progress = []
    LEVEL_WIN_DIST = 3161
    
    for t in count():

        if t % 1000 == 0:
            print('Timestep:', t)

        ### Step the env and store the transition
        last_idx = replay_buffer.store_frame(last_obs)

        recent_observations = replay_buffer.encode_recent_observation()

        recent_hot_actions = replay_buffer.encode_recent_hot_actions()


        if t > learning_starts:
            action = select_epilson_greedy_action(Q, recent_hot_actions, recent_observations, last_obs, t)[0]
        else:
            action = random.randrange(num_actions)
            
        # one hot encoding
        act_onehot = to_onehot(action, num_actions)
        replay_buffer.store_hot_action(last_idx, act_onehot)

        obs, reward, done, _info = env.step(act_onehot)
        #reward = max(-1.0, min(reward, 1.0))

        replay_buffer.store_effect(last_idx, action, reward, done)
        
        if done:
            obs = env.reset()
            num_trials += 1
            x_pos = _info['x_pos']
            percent_done = (x_pos / LEVEL_WIN_DIST) * 100.
            trial_progress.append(x_pos)
            print("Completed trial {0} @ [{1}/{2}]({3:.2f}%)".format(num_trials, x_pos, LEVEL_WIN_DIST, percent_done))

        last_obs = obs
        
        if (t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size)):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, hot_act_batch, next_hot_act_batch = replay_buffer.sample(batch_size)
            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)

            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)
            hot_act_batch = Variable(torch.from_numpy(hot_act_batch)).type(dtype)
            next_hot_act_batch = Variable(torch.from_numpy(next_hot_act_batch)).type(dtype)


            last_obs_reshape = last_obs.reshape(-1, 84, 84)
            last_obs_batch = np.concatenate([last_obs_reshape.copy() for i in range(batch_size)], 0)
            last_obs_batch = Variable(torch.from_numpy(last_obs_batch)).type(dtype)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()

            if model == "DQN":
                current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze(1)
                next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
            else:
                current_Q_values = Q(hot_act_batch, obs_batch, last_obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze(1)
                next_max_q = target_Q(next_hot_act_batch, next_obs_batch, last_obs_batch).detach().max(1)[0]

            next_Q_values = not_done_mask * next_max_q

            # TD value
            target_Q_values = rew_batch + (gamma * next_Q_values)

            # Compute Bellman error
            bellman_error = target_Q_values - current_Q_values
            # clip the bellman error between [-1, 1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # * -1 gradient，why?
            d_error = clipped_bellman_error * -1.0

            if t % 1000 == 0:
                loss_arr.append(d_error.mean())

            # backward & update
            optimizer.zero_grad()
            current_Q_values.backward(d_error.data)
            
            optimizer.step()
            num_param_updates += 1
            
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())
                
        ### Log & track
        episode_rewards = env.get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:]) 

        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
            
        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("> Timestep: %d" % (t,))
            print("> Mean reward (100 episodes): %f" % mean_episode_reward)
            print("> Best mean reward: %f" % best_mean_episode_reward)
            print("> Trials: %d" % len(episode_rewards))
            print("> Exploration value: %f" % exploration.value(t))
            
            reward_arr.append(mean_episode_reward)

            sys.stdout.flush()
            
            # Save the trained model
            torch.save(Q.state_dict(), 'mario_Q_params.pkl')
            torch.save(target_Q.state_dict(), 'mario_target_Q_params.pkl')

            with open('statistics/trial_progress.pkl', 'wb') as f:
                pickle.dump(trial_progress, f)

            with open('statistics/loss_arr.pkl', 'wb') as f:
                pickle.dump(loss_arr, f)

            with open('statistics/reward_arr.pkl', 'wb') as f:
                pickle.dump(reward_arr, f)
