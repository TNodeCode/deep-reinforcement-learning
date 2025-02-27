import torch
import torch.distributions as distributions
import numpy as np


class AgentUtils:
    """Class that provides helper methods for agents.
    """
    @staticmethod
    def random_discrete_action(probs: torch.Tensor) -> np.ndarray:
        """Choose a random discrete action based on an arary of probabilities.

        Args:
            probs (torch.Tensor): 2D array (shape of n_agents x action probabilities) that represents probabilities for choosing an action for each agent

        Returns:
            np.ndarray: 1D Array (shape n_agents) containing chosen actions
        """
        probs = probs.cpu().detach().numpy()
        actions = []
        for prob in probs:
            action = np.random.choice(prob.shape[0], p=prob)
            actions.append(action)
        return np.array(actions)
    
    @staticmethod
    def random_normal_action(means: torch.Tensor, stds: torch.Tensor) -> np.ndarray:
        """Choose a random action by sampling from a normal distribution.

        Args:
            means (torch.Tensor): tensor of shape (n_agents, dim_action) containing means
            stds (torch.Tensor): tensor of shape (n_agents, dim_action) containing standard deviations

        Returns:
            tuple of:
            np.ndrray: array of shape (n_agents, dim_action) containing chosen actions
            np.ndrray: array of shape (n_agents) log probabilities for actions
        """
        actions = []
        log_probs = []
        for mean, std in zip(means, stds):
            normal_dist = distributions.Normal(mean, std)
            action = normal_dist.sample()
            normal_dist.log_prob(action)
            actions.append(action.cpu().numpy())
        return np.array(actions), np.array(log_probs)
