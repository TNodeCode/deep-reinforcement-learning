import torch
import torch.nn as nn


class DeepConvQNetwork(nn.Module):
    """
    This network maps state information to actions. It is a simple feed-forward neural network.
    """
    def __init__(
            self,
            state_shape,
            n_actions,
            conv_channels=[16,32,64],
            hidden_dims=[64],
            activation=nn.LeakyReLU,
            norm=nn.LayerNorm,
            seed=42
        ):
        """
        Constructor.
        
        Args:
            state_shape (tuple): shape of state (H, W, C)
            n_actions (int): number of possible actions
            conv_channels (list[int]): number of channels in the convolutional feature extractor
            hidden_dims (list[int]): hidden dimensions of neural network
            seed (int): Random seed
        """
        super(DeepConvQNetwork, self).__init__()
        # It is important that both networks use the same seed as they should be copies of each other in the beginning
        self.seed = torch.manual_seed(seed)
        self.state_shape = state_shape
        H, W, C = state_shape
        self.n_actions = n_actions
        channels_featex = [C] + conv_channels
        n_layers = len(channels_featex) - 1
        # the feature extractor consumes the image and uses convolutional layers to build a feature tensor
        self.feature_extractor = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=3, stride=1, padding=1),
                activation(),
                norm([dim_out, H // 2**i, W // 2**i]),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
            for i, (dim_in, dim_out) in enumerate(zip(channels_featex[:-1], channels_featex[1:]))
        ])
        # the head transforms the feature tensor into score values for actions
        # the last layer needs to be linear and not followed by an activation function
        n_features = (W // 2**n_layers) * (H // 2**n_layers) * conv_channels[-1]
        dim_head = [n_features] + hidden_dims
        self.head = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                activation(),
            )
            for dim_in, dim_out in zip(dim_head[:-1], dim_head[1:])
        ], nn.Linear(dim_head[-1], n_actions))
        
    def forward(self, x):
        """
        Run state vector through network and get action vector.
        
        Args:
            x (1D vector): states vector
            
        Returns:
            1D vector: scores for possible actions to take
        """
        x = torch.permute(x, (0,3,1,2))
        return self.head(torch.flatten(self.feature_extractor(x), start_dim=1, end_dim=-1))
    
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