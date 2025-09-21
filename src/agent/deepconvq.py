import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from src.agent.deepq import DeepQAgent
from src.network.q_conv_network import DeepConvQNetwork
from src.network.utils import NetworkUtils


class DeepConvQAgent(DeepQAgent):
    def __init__(
            self,
            env,
            eps=1.0,
            lr=5e-4,
            conv_channels=[3,16,32,64],
            hidden_dims=[64],
            memory_size=int(1e5),
            activation=nn.LeakyReLU,
            batch_size=256,
            max_steps=100,
            update_every=10,
            device="cpu"
        ):
        """
        Constructor.
        """
        super().__init__(
            env,
            eps=eps,
            lr=lr,
            hidden_dims=hidden_dims,
            memory_size=memory_size,
            activation=activation,
            batch_size=batch_size,
            max_steps=max_steps,
            update_every=update_every,
            device=device
        )
        self.eps = eps
        self.gamma = torch.tensor(0.99).to(device)
        self.tau = 1e-3
        self.update_every = update_every
        self.device = device
        # Build two Deep-Q-Networks 
        self.local_q_network = DeepConvQNetwork(
            state_shape=(224,224,1),
            n_actions=self.env.get_num_actions(),
            conv_channels=conv_channels,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)        
        self.target_q_network = DeepConvQNetwork(
            state_shape=(224,224,1),
            n_actions=self.env.get_num_actions(),
            conv_channels=conv_channels,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.local_q_network.parameters(), lr=lr)
        