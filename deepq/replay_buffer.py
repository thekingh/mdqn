import numpy as np
import random

def sample_n_unique(sampling_f, n):
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)

    return res

class ReplayBuffer(object):
    
    def __init__(self, capacity, frame_history_len, num_actions):   
        self.capacity = capacity 
        self.frame_history_len = frame_history_len
        self.num_actions = num_actions

        self.next_idx = 0
        self.size     = 0

        self.obs    = None
        self.action = None
        self.reward = None
        self.done   = None
        self.hot_actions = None

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.size

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        next_obs_batch = np.concatenate([self._encode_observation(idx+1)[np.newaxis, :] for idx in idxes], 0)
        action_batch = self.action[idxes]
        reward_batch = self.reward[idxes]
        done_mask    = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        hot_actions_batch = np.concatenate([self._encode_hot_action(idx)[np.newaxis, :] for idx in idxes], 0)
        next_hot_actions_batch = np.concatenate([self._encode_hot_action(idx+1)[np.newaxis, :] for idx in idxes], 0)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_mask, hot_actions_batch, next_hot_actions_batch
    
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.size - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_hot_actions(self):
        assert self.size > 0
        return self._encode_hot_action((self.next_idx - 1) % self.capacity)

    def _encode_hot_action(self, idx):
        end_idx   = idx + 1
        start_idx = end_idx - self.frame_history_len

        if start_idx < 0 and self.size != self.capacity: start_idx = 0

        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.capacity]:
                start_idx = idx + 1

        missing_context = self.frame_history_len - (end_idx - start_idx)
        if start_idx < 0 or missing_context > 0:
            hot_actions = [np.zeros_like(self.hot_actions[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                hot_actions.append(self.hot_actions[idx % self.capacity])
            return np.concatenate(hot_actions, 0)
        else:
            return self.hot_actions[start_idx:end_idx].reshape(-1, self.num_actions)

    def encode_recent_observation(self):
        assert self.size > 0
        return self._encode_observation((self.next_idx - 1) % self.capacity)

    def _encode_observation(self, idx):
        end_idx   = idx + 1
        start_idx = end_idx - self.frame_history_len

        # check if low dimensional obs (such as RAM)
        if len(self.obs.shape) == 2: return self.obs[end_idx-1]

        if start_idx < 0 and self.size != self.capacity: start_idx = 0

        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.capacity]:
                start_idx = idx + 1

        missing_context = self.frame_history_len - (end_idx - start_idx)
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]

            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.capacity])

            return np.concatenate(frames, 0)
        else:
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_h)

    def store_frame(self, frame):
        if len(frame.shape) > 1:
            frame = frame.transpose(2, 0, 1)
        if self.obs is None:
            self.obs    = np.empty([self.capacity] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.capacity],                     dtype=np.int32)
            self.reward = np.empty([self.capacity],                     dtype=np.float32)
            self.done   = np.empty([self.capacity],                     dtype=np.bool)
            self.hot_actions = np.empty([self.capacity] + list((1, self.num_actions)), dtype=np.int32)

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        return ret

    def store_hot_action(self, idx, action):
        self.hot_actions[idx] = action

    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done
