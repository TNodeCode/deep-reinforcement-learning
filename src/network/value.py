import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, n_states, hidden_dims=[64], activation=nn.ReLU, norm=nn.LayerNorm):
        """
        Initialize the value network.

        Args:
            n_states: Number of input features (state dimensions).
            hidden_dims: List of integers defining the number of neurons in each hidden layer.
            activation: Activation function to use between layers.
        """
        super(ValueNetwork, self).__init__()
        
        # Define the network architecture
        layers = []
        input_dim = n_states
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            layers.append(norm(hidden_dim))
            input_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(input_dim, 1))  # Output a single value
        
        # Compile the layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor (state representation).

        Returns:
            Tensor containing the estimated value.
        """
        return self.model(x)