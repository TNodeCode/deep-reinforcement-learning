import torch
import torch.nn as nn

class PolicyContinuousNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dims=[64], activation=nn.LeakyReLU):
        """
        Initialize the policy network.

        Args:
            n_states: Number of input features (state dimensions).
            n_actions: Number of possible actions.
            hidden_dims: List of integers defining the number of neurons in each hidden layer.
            activation: Activation function to use between layers.
        """
        super(PolicyContinuousNetwork, self).__init__()
        
        # Define the network architecture
        layers = []
        input_dim = n_states
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        # Compile the layers into a sequential model
        self.model = nn.Sequential(*layers)
        self.head_mean = nn.Sequential(
            nn.Linear(hidden_dim, n_actions),
        )
        self.head_std = nn.Sequential(
            nn.Linear(hidden_dim, n_actions),
            nn.Softplus(),  # Ensures strictly positive output
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor (state representation).

        Returns:
            Tensor of action scores.
        """
        x = self.model(x)
        mean, std = self.head_mean(x), self.head_std(x)
        return mean, torch.clamp(std, min=1e-3)