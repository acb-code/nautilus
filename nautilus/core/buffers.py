from __future__ import annotations
from dataclasses import dataclass
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, action_shape):
        self.capacity = int(capacity)
        self.idx = 0
        self.full = False
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        high = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, high, size=batch_size)
        return (
            self.obs[idxs], self.actions[idxs], self.rewards[idxs],
            self.next_obs[idxs], self.dones[idxs]
        )
