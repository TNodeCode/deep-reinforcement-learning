import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dims=[64], activation=nn.LeakyReLU):
        """
        Initialize the policy network.

        Args:
            n_states: Number of input features (state dimensions).
            n_actions: Number of possible actions.
            hidden_dims: List of integers defining the number of neurons in each hidden layer.
            activation: Activation function to use between layers.
        """
        super(PolicyNetwork, self).__init__()
        
        # Define the network architecture
        layers = []
        input_dim = n_states
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(input_dim, n_actions))
        layers.append(nn.Softmax(dim=-1))
        
        # Compile the layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor (state representation).

        Returns:
            Tensor of action scores.
        """
        return self.model(x)