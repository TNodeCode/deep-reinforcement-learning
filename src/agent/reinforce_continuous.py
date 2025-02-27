import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from src.agent.reinforce import ReinforceAgent 
from src.network.policy_continuous import PolicyContinuousNetwork

class ReinforceContinuousAgent(ReinforceAgent):
    def __init__(
            self,
            env,
            lr=1e-3,
            gamma=0.99,
            hidden_dims=[64],
            max_steps=100,
            device="cpu",
            **kwargs,
        ):
        """
        Constructor for Continuous REINFORCE agent.

        Args:
            env: The environment to interact with
            lr: Learning rate for the policy network
            gamma: Discount factor for rewards
            hidden_dims: List of hidden layer sizes for the policy network
            device: Device to use for computation ('cpu' or 'cuda')
        """
        super().__init__(env, lr=lr, gamma=gamma, hidden_dims=hidden_dims, max_steps=max_steps, device=device, **kwargs)
        
        # Redefine policy network output layer for continuous actions
        self.policy_network = PolicyContinuousNetwork(
            n_states=self.env.get_state_shape()[-1],
            n_actions=self.env.get_action_shape()[-1],
            hidden_dims=hidden_dims
        ).to(device)

    def choose_action(self):
        """
        Choose an action based on the current policy using a Gaussian distribution.

        Returns:
            np.ndarray: Action chosen
        """
        state_tensor = torch.from_numpy(self.normalize_state(self.state)).float().to(self.device)
        mean, std = self.policy_network(state_tensor)
        actions = []
        for _mean, _std in zip(mean, std):
            # create a normal distribution based on the policy output and sample from that distribution
            dist = torch.distributions.Normal(_mean, _std)
            actions.append(dist.sample().cpu().numpy())
        return np.array(actions)

    def learn(self):
        """
        Learn from a single episode for continuous action spaces.
        """
        returns = self.calculate_returns([reward for _, _, reward in self.episode])
        
        self.optimizer.zero_grad()
        for (state, action, _), Gt in zip(self.episode, returns):
            state_tensor = torch.from_numpy(self.normalize_state(state)).float().to(self.device)
            action_tensor = torch.tensor(action, dtype=torch.float).to(self.device)
            mean, std = self.policy_network(state_tensor)
            normal_dist = torch.distributions.Normal(mean, std)
            log_prob = normal_dist.log_prob(action_tensor).sum()
            entropy = normal_dist.entropy()
            loss = -log_prob * torch.tensor(Gt, device=self.device) - self.agent_params["entropy_factor"] * entropy.mean()
            loss.backward()
        
        self.optimizer.step()