import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from src.agent.simple import SimpleAgent
from src.agent.utils import AgentUtils
from src.network.policy_continuous import PolicyContinuousNetwork
from src.network.value import ValueNetwork

class A2CContinuousAgent(SimpleAgent):
    def __init__(
            self,
            env,
            lr=1e-4,
            gamma=0.99,
            hidden_dims=[64],
            device="cpu"
        ):
        """
        Constructor for A2C agent with continuous action space.

        Args:
            env: The environment to interact with.
            lr: Learning rate for both policy and value networks.
            gamma: Discount factor for rewards.
            hidden_dims: List of hidden layer sizes for both networks.
            device: Device to use for computation ('cpu' or 'cuda').
        """
        super().__init__(env, device=device)
        
        self.gamma = gamma
        self.device = device
        
        # Actor network for continuous actions
        self.policy_network = PolicyContinuousNetwork(
            n_states=self.env.get_state_shape()[-1],
            n_actions=self.env.get_action_shape()[-1],  # Mean and log_std for each action dimension
            hidden_dims=hidden_dims
        ).to(device)

        # Critic network
        self.value_network = ValueNetwork(
            n_states=self.env.get_state_shape()[-1],
            hidden_dims=hidden_dims
        ).to(device)

        # Optimizers
        self.policy_optimizer = optim.AdamW(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.AdamW(self.value_network.parameters(), lr=lr)

    def choose_action(self):
        """
        Choose an action based on the current policy using a Gaussian distribution.

        Returns:
            np.ndarray: Action chosen.
        """
        state_tensor = torch.from_numpy(self.state).float().to(self.device)
        means, log_stds = self.policy_network(state_tensor)
        stds = torch.exp(log_stds)
        actions, log_probs = AgentUtils.random_normal_action(means=means, stds=stds)
        return actions

    def play(self):
        """
        Play a single episode and gather trajectories.

        Returns:
            float: episode score
        """
        episode = []
        state = self.env.get_current_state()
        for _ in range(self.max_steps):
            action = self.choose_action()
            state_before, reward, state_after, done = self.do_step(action)
            episode.append((state_before, action, reward, state_after, done))
            if all(done):
                break
        return self.score

    def learn(self):
        """
        Learn from a single episode using Advantage Actor-Critic approach.
        """
        episode = self.play()
        
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        for (state, action, reward, next_state, done) in episode:
            state_tensor = torch.from_numpy(state).float().to(self.device)
            next_state_tensor = torch.from_numpy(next_state).float().to(self.device)
            action_tensor = torch.tensor(action, dtype=torch.float).to(self.device)

            # Calculate value estimates
            value = self.value_network(state_tensor)
            next_value = self.value_network(next_state_tensor)
            advantage = reward + self.gamma * next_value * (1 - done) - value

            # Actor loss (policy gradient with advantage)
            means, log_stds = self.policy_network(state_tensor)
            stds = torch.exp(log_stds)
            actions, log_probs = AgentUtils.random_normal_action(means=means, stds=stds)
            log_prob = log_probs.sum(dim=-1)
            actor_loss = -log_prob * advantage.detach()

            # Critic loss (value function update)
            critic_loss = advantage.pow(2)

            # Backpropagation
            actor_loss.backward()
            critic_loss.backward()

        # Update networks
        self.policy_optimizer.step()
        self.value_optimizer.step()