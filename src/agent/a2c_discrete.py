import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from src.agent.simple import SimpleAgent
from src.agent.utils import AgentUtils
from src.network.policy import PolicyNetwork
from src.network.value import ValueNetwork

class A2CDiscreteAgent(SimpleAgent):
    def __init__(
            self,
            env,
            lr=1e-4,
            gamma=0.99,
            hidden_dims=[64],
            device="cpu",
            **kwargs
        ):
        """
        Constructor for A2C agent.

        Args:
            env: The environment to interact with.
            lr: Learning rate for both policy and value networks.
            gamma: Discount factor for rewards.
            hidden_dims: List of hidden layer sizes for both networks.
            device: Device to use for computation ('cpu' or 'cuda').
        """
        super().__init__(env, device=device, **kwargs)
        
        self.gamma = gamma
        self.device = device
        self.n_agents = kwargs["n_agents"] if "n_agents" in kwargs else 1
        
        # Actor network
        self.policy_network = PolicyNetwork(
            n_states=self.env.get_state_shape()[-1],
            n_actions=self.env.get_num_actions(),
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
        Choose an action based on the current policy.

        Returns:
            np.ndarray: Action chosen.
        """
        state_tensor = torch.from_numpy(self.state).float().to(self.device)
        probs = self.policy_network(state_tensor)
        return AgentUtils.random_discrete_action(probs=probs)

    def play(self):
        """
        Play a single episode and gather trajectories.

        Returns:
            float: episode score
        """
        self.episodes = []
        self.scores = []
        for n in range(self.n_agents):
            self.episode = []
            self.reset()
            for _ in range(self.max_steps):
                action = self.choose_action()
                state_before, reward, state_after, done = self.do_step(action)
                self.episode.append((state_before, action, reward, state_after, done))
                if all(done):
                    break
            self.episodes.append(self.episode)
            self.scores.append(self.score)
        self.learn()
        return sum(self.scores) / len(self.scores)

    def learn(self):
        """
        Learn from a single episode using Advantage Actor-Critic approach.
        """
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        # Iterate over episodes that were produced by the agents
        for episode in self.episodes:
            # Initialize global loss for all agents
            actor_loss = torch.tensor([[0.0]])
            critic_loss = torch.tensor([[0.0]])
            for (state, action, reward, next_state, done) in episode:
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
                log_prob = F.log_softmax(self.policy_network(state_tensor), dim=-1)[0, action_tensor]
                actor_loss += -log_prob * advantage.detach()

                # Critic loss (value function update)
                critic_loss += advantage.pow(2)

            # Backpropagation over mean losses
            (actor_loss / len(self.episodes)).backward()
            (critic_loss / len(self.episodes)).backward()

        # Update networks
        self.policy_optimizer.step()
        self.value_optimizer.step()