
import torch 
import numpy as np 


class ReplayMemory:
    """ Observations will be stored and retrieved in NHWC format """
    
    def __init__(self, mem_size, obs_shape):
        self.ptr = 0
        self.filled = 0
        self.mem_size = mem_size 
        
        self.obses = np.zeros((self.mem_size, *obs_shape), dtype=np.uint8)
        self.next_obses = np.zeros((self.mem_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((self.mem_size,), dtype=np.uint8)
        self.rewards = np.zeros((self.mem_size), dtype=np.float32)
        self.dones = np.zeros((self.mem_size,), dtype=np.uint8)
        
    def store(self, obs, action, reward, next_obs, done):
        self.obses[self.ptr] = np.array(obs)
        self.next_obses[self.ptr] = np.array(next_obs)
        self.actions[self.ptr] = action 
        self.rewards[self.ptr] = reward 
        self.dones[self.ptr] = int(done) 
        
        self.ptr = (self.ptr + 1) % self.mem_size 
        self.filled = min(self.filled + 1, self.mem_size)
        
    def get_batch(self, batch_size):
        assert self.filled > batch_size, 'Not enough samples in memory yet'
        idx = np.random.choice(np.arange(self.filled), size=batch_size, replace=False)
        
        obs = self.obses[idx]
        next_obs = self.next_obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        done = self.dones[idx]
        return (obs, action, reward, next_obs, done)