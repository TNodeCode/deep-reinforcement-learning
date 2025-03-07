import torch
import random
import numpy as np
from collections import deque



class Memory:
    def __init__(self, maxlen, device="cpu", action_space_discrete=True):
        """
        Initializes the replay buffer.

        Parameters:
            max_size: The maximum number of experiences the buffer can hold
        """
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.state_dtype = torch.float32
        self.action_dtype = torch.long if action_space_discrete else torch.float32
        self.device=device

    def append(self, states, actions, rewards, next_states, dones):
        """
        Adds a new experience to the buffer.

        Parameters:
            state: The initial state
            action: The action taken
            reward: The reward received
            next_state: The state after taking the action
            done: A boolean indicating if the episode ended
        """
        for i in range(len(states)):
            experience = (states[i], actions[i], rewards[i], next_states[i], dones[i])
            self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Parameters:
            batch_size: The number of experiences to sample

        Returns:
            A tuple of batch tensors: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=self.state_dtype, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=self.action_dtype, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=self.state_dtype, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)