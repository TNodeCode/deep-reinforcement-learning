import torch
import torch.nn as nn


class DeepQNetwork(nn.Module):
    """
    This network maps state information to actions. It is a simple feed-forward neural network.
    """
    def __init__(self, n_states, n_actions, hidden_dims=[64], activation=nn.LeakyReLU, norm=nn.LayerNorm, seed=42):
        """
        Constructor.
        
        Args:
            n_states (int): number of states
            n_actions (int): number of possible actions
            hidden_dims (list[int]): hidden dimensions of neural network
            seed (int): Random seed
        """
        super(DeepQNetwork, self).__init__()
        # It is important that both networks use the same seed as they should be copies of each other in the beginning
        self.seed = torch.manual_seed(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        dim_layers = [n_states] + hidden_dims
        # constructing q-network from blocks of linear layers, activation layers and normalization layers.
        # the last layer needs to be a linear layer without an activation layer following it.
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                activation(),
                norm(dim_out),
            )
            for dim_in, dim_out in zip(dim_layers[:-1], dim_layers[1:])
        ], nn.Linear(dim_layers[-1], n_actions))
        
    def forward(self, x):
        """
        Run state vector through network and get action vector.
        
        Args:
            x (1D vector): states vector
            
        Returns:
            1D vector: scores for possible actions to take
        """
        assert len(x.shape) == 2, "Input tensor should be two-dimensional"
        assert x.shape[1] == self.n_states, f"Input vectors dimension should match size of state vectors ({self.n_states}), given tensor is of shape {x.shape}"
        return self.net(x)
    
    def get_best_choice(self, actions):
        """
        Get best choice from action scores.
        
        Args:
            actions (1D vector): Vector that contains scores for each action
            
        Returns:
            int: Index of best action
        """
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)
        max_values, max_indices = torch.max(actions, dim=1)
        return max_indices, max_values