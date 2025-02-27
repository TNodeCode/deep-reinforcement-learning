import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.agent.simple import SimpleAgent
from src.agent.utils import AgentUtils
from src.network.policy import PolicyNetwork
from src.network.value import ValueNetwork

class A3CAgent(SimpleAgent):
    def __init__(
            self,
            env,
            lr=1e-4,
            gamma=0.99,
            beta=0.01,
            hidden_dims=[64],
            max_steps=100,
            device="cpu",
            **kwargs
        ):
        """
        Constructor for A3C agent.

        Args:
            env: The environment to interact with
            lr: Learning rate for both the policy and value networks
            gamma: Discount factor for rewards
            beta: Entropy regularization factor
            hidden_dims: List of hidden layer sizes for the networks
            device: Device to use for computation ('cpu' or 'cuda')
        """
        super().__init__(env, device=device, action_space_discrete=True, max_steps=max_steps, **kwargs)
        self.gamma = gamma
        self.beta = beta
        self.device = device
        
        # Build policy and value networks
        self.policy_network = PolicyNetwork(
            n_states=self.env.get_state_shape()[-1],
            n_actions=self.env.get_num_actions(),
            hidden_dims=hidden_dims
        ).to(device)
        
        self.value_network = ValueNetwork(
            n_states=self.env.get_state_shape()[-1],
            hidden_dims=hidden_dims
        ).to(device)

        self.optimizer = optim.AdamW(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()), lr=lr
        )

    def choose_action(self):
        """
        Choose an action based on the current policy.

        Returns:
            np.ndarray: Action chosen
        """
        state_tensor = torch.from_numpy(self.state).float().to(self.device)
        probs = self.policy_network(state_tensor)
        return AgentUtils.random_discrete_action(probs=probs)

    def learn(self):
        """
        Learn from a single episode using the A3C approach.
        """
        self.optimizer.zero_grad()
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        for (state, action, reward, next_state, done) in self.memory:
            state_tensor = torch.from_numpy(state).float().to(self.device)
            next_state_tensor = torch.from_numpy(next_state).float().to(self.device)
            action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            done = torch.tensor(done, dtype=torch.float32).to(self.device)

            # Calculate value estimates
            value = self.value_network(state_tensor)
            next_value = self.value_network(next_state_tensor)
            advantage = reward + self.gamma * next_value * (1 - done) - value

            # Actor loss (policy gradient with advantage)
            log_prob = torch.log(self.policy_network(state_tensor)[0, action_tensor])
            policy_loss += -log_prob * advantage.detach()

            # Critic loss (value function update)
            value_loss += advantage.pow(2)

            # Entropy loss for exploration
            entropy_loss += -torch.sum(self.policy_network(state_tensor) * torch.log(self.policy_network(state_tensor)))

        # Combine losses
        total_loss = policy_loss + 0.5 * value_loss - self.beta * entropy_loss
        total_loss.backward()
        self.optimizer.step()

    def play(self):
        """
        Play a single episode and gather trajectories.
        
        Returns:
            float: Episode score
        """
        self.memory = []
        self.score = 0
        self.reset()
        for _ in range(self.max_steps):
            action = self.choose_action()
            state_before, reward, state_after, done = self.do_step(action)
            self.memory.append((state_before, action, reward, state_after, done))
            self.score += reward
            if all(done):
                break
        self.learn()
        return self.score
    
    def save(self, dir="."):
        """
        Save knowledge of agent.
        """
        torch.save(self.policy_network.state_dict(), os.path.join(dir, 'a3c_policy.pth'))
        torch.save(self.value_network.state_dict(), os.path.join(dir, 'a3c_value.pth'))

    def load(self, dir="."):
        """
        Load agent knowledge.

        Args:
            dir (str): Directory where knowledge (weight files) is stored.
        """
        self.policy_network.load_state_dict(torch.load(os.path.join(dir, 'a3c_policy.pth')))
        self.value_network.load_state_dict(torch.load(os.path.join(dir, 'a3c_value.pth')))