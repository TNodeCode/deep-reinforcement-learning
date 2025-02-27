import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from src.agent.simple import SimpleAgent
from src.network.policy import PolicyNetwork  # Assume you have a similar network class

class ReinforceAgent(SimpleAgent):
    def __init__(
            self,
            env,
            lr=1e-3,
            gamma=0.99,
            hidden_dims=[64],
            max_steps=100,
            device="cpu",
            **kwargs
        ):
        """
        Constructor for REINFORCE agent.

        Args:
            env: The environment to interact with
            lr: Learning rate for the policy network
            gamma: Discount factor for rewards
            hidden_dims: List of hidden layer sizes for the policy network
            device: Device to use for computation ('cpu' or 'cuda')
        """
        super().__init__(env, device=device, action_space_discrete=True, max_steps=max_steps, **kwargs)
        self.gamma = gamma
        self.device = device
        
        # Build policy network
        self.policy_network = PolicyNetwork(
            n_states=self.env.get_state_shape()[-1],
            n_actions=self.env.get_action_shape()[-1] if self.env.is_action_space_continuous() else self.env.get_num_actions(),
            hidden_dims=hidden_dims
        ).to(device)

        self.optimizer = optim.AdamW(self.policy_network.parameters(), lr=lr)

    def choose_action(self):
        """
        Choose an action based on the current policy.

        Returns:
            np.ndarray: Action chosen
        """
        state_tensor = torch.from_numpy(self.normalize_state(self.state)).float().to(self.device)
        probs = self.policy_network(state_tensor)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(self.env.get_num_actions(), p=prob.cpu().detach().numpy()))
        return np.array([actions])
    
    def play(self):
        """
        Play a single episode and gather trajectories.
        
        Returns:
            float: Episode score
        """
        self.episode = []
        self.score = np.array([0.0])
        for _ in range(self.max_steps):
            action = self.choose_action()
            if not self.env.is_action_space_continuous():
                action = action.squeeze(axis=0)
            state_before, reward, state_after, done = self.do_step(action)
            self.episode.append((self.normalize_state(state_before), action, reward))
            self.score += reward
            if all(done):
                break
        self.learn()
        return self.score

    def learn(self):
        """
        Learn from a single episode.
        """
        returns = self.calculate_returns([reward for _, _, reward in self.episode])
        self.optimizer.zero_grad()
        for (state, action, _), Gt in zip(self.episode, returns):
            state_tensor = torch.from_numpy(self.normalize_state(state)).float().to(self.device)
            action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)
            log_prob = torch.log(self.policy_network(state_tensor))[0, action_tensor]
            loss = -log_prob * torch.tensor(Gt, device=self.device)
            loss.backward()
        self.optimizer.step()

    def calculate_returns(self, rewards):
        """
        Calculate the returns for each timestep in the episode.

        Args:
            rewards (list): List of rewards for the episode.

        Returns:
            list: List of returns for each timestep.
        """
        mean, std = np.mean(rewards), np.std(rewards) + 1e-9
        
        returns = []
        Gt = 0
        for reward in reversed(rewards):
            Gt = (reward - mean) / std + self.gamma * Gt
            returns.insert(0, Gt)
        return returns
    
    def normalize_state(self, state):
        """Normalize the state.

        Args:
            state (np.ndarray): State vector

        Returns:
            np.ndarray: Normalized state vector
        """
        state[np.isnan(state)] = 0
        state = np.clip(state, -1e8, 1e8)
        return state

    def save(self, dir="."):
        """
        Save knowledge of agent.
        """
        # save weights of local network
        torch.save(self.policy_network.state_dict(), os.path.join(dir, 'reinforce.pth'))

    def load(self, dir="."):
        """
        Load agent knowledge.

        Args:
            dir (str): Directory where knowledge (weigh files) is stored.
        """
        self.policy_network.load_state_dict(torch.load(os.path.join(dir, 'reinforce.pth')))