import os
import random
import logging
import numpy as np
#from utils import save_npy, load_npy

class ReplayMemory:
    def __init__(self, model_dir):
        self.model_dir = model_dir        
        self.memory_size = 1000000
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float64)
        self.prestate = np.empty((self.memory_size, 82), dtype = np.float16)
        self.poststate = np.empty((self.memory_size, 82), dtype = np.float16)
        self.batch_size = 2000
        self.count = 0
        self.current = 0
        

    def add(self, prestate, poststate, reward, action):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size
        
   
           
    def sample(self):
        indexes = []
        while len(indexes) < self.batch_size:
            index = random.randint(0, self.count - 1)
            indexes.append(index)
        prestate = self.prestate[indexes]
        poststate = self.poststate[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        return prestate, poststate, actions, rewards
   
