import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from src.environment.abstract import AbstractEnvironment
from src.agent.simple import SimpleAgent
from src.network.q_network import DeepQNetwork
from src.network.utils import NetworkUtils


class DeepQAgent(SimpleAgent):
    """This class is an implementation of an agent interacting with an environment that is describes by a continuous 1D vector.
    The agent implementes the Deep Q Learning Algorithm for training its internal networks.
    """
    def __init__(
            self,
            env: AbstractEnvironment,
            eps: float=1.0,
            lr: float=5e-4,
            hidden_dims: list[int]=[64],
            memory_size: int=int(1e5),
            activation: nn.Module=nn.LeakyReLU,
            batch_size: int=256,
            max_steps: int=100,
            update_every: int=10,
            clip_grad_norm:float=-1,
            device: str="cpu"
        ):
        """_summary_

        Args:
            env (AbstractEnvironment): instance of AbstractEnvironment
            eps (float, optional): Start value for epsilon value for greedy strategy. Defaults to 1.0.
            lr (float, optional): learning rate. Defaults to 5e-4.
            hidden_dims (list[int], optional): hidden dimensions of q-network. Defaults to [64].
            memory_size (int, optional): size of replay buffer. Defaults to int(1e5).
            activation (nn.Module, optional): Activation function for q-network. Defaults to nn.LeakyReLU.
            batch_size (int, optional): batch size. Defaults to 256.
            max_steps (int, optional): maximum steps in environment. Defaults to 100.
            update_every (int, optional): update target network every x steps. Defaults to 10.
            clip_grad_norm (float, optional): gradient clipping value. Defaults to -1.
            device (str, optional): computation device. Defaults to "cpu".
        """
        super().__init__(
            env,
            eps=eps,
            memory_size=memory_size,
            action_space_discrete=True,
            batch_size=batch_size,
            max_steps=max_steps,
            device=device
        )
        self.eps = eps
        self.gamma = torch.tensor(0.99).to(device)
        self.tau = 1e-3
        self.update_every = update_every
        self.clip_grad_norm=clip_grad_norm
        self.device = device
        # Build two Deep-Q-Networks 
        self.local_q_network = DeepQNetwork(
            n_states=self.env.get_state_shape()[-1],
            n_actions=self.env.get_num_actions(),
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)        
        self.target_q_network = DeepQNetwork(
            n_states=self.env.get_state_shape()[-1],
            n_actions=self.env.get_num_actions(),
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.local_q_network.parameters(), lr=lr)
        
    def choose_action(self):
        """
        Choose an action based on internal knowledge.
        """
        # For choosing next step put local_q_network into validation mode
        self.local_q_network.eval()
        # Use local_q_network to compute scores for each action based on state and get best action
        with torch.no_grad():
            # compute scores for each action
            state_tensor = torch.from_numpy(self.state).float()
            state_tensor = state_tensor.to(self.device)
            action_scores = self.local_q_network(state_tensor)
            # get index of best action
            best_action_idx, _ = self.local_q_network.get_best_choice(action_scores)
        # Put local_q_network back into training mode
        self.local_q_network.train()
        # Epsilon greedy strategy
        if random.random() > self.eps:
            return np.array([int(best_action_idx)])
        else:
            return np.array([np.random.randint(self.env.get_num_actions())])
        
    def learn(self):
        """Update the networks.
        """
        # first there need to be enough tuples in the memory before we can sample from it.
        if len(self.memory) < self.batch_size:
            return
        
        # check whether we want to do an update now
        if self.current_step % self.update_every != 0:
            return

        # states: shape [batch_size, state_dim]
        # actions: shape [batch_size] containing indices of chosen actions
        # rewards: shape [batch_size, 1]
        # next states: shape[batch_size, state_dim]
        # dones: shape[batch_size, 1]
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Calculate target Q-values
        # target_q_values: shape [batch_size, 1] containing target scores for chosen actions
        with torch.no_grad():
            max_next_q_values = self.target_q_network(next_states).max(dim=1, keepdim=True)[0] # shape [batch_size, 1]
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Calculate current Q-values
        # network outputs tensor of shape [batch_size, n_actions]

        # compute scores for each action
        all_actions_scores = self.local_q_network(states) # shape [batch_size, n_actions]

        # gather scores of actions that were actually chosen. Gather along action axis (axis 1) and collect scores indicated by the action tensor which contains the indices.
        actions = actions.unsqueeze(axis=1) # shape [batch_size, 1]
        current_q_values = all_actions_scores.gather(1, actions) # shape [batch_size, 1]

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the local Q-network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Optional: Gradient clipping
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.local_q_network.parameters(), self.clip_grad_norm)

        # Update weights
        self.optimizer.step()

        # Update target network with soft updates
        NetworkUtils.soft_update(self.local_q_network, self.target_q_network, tau=self.tau)
        
    def save(self, dir="./output"):
        """
        Save knowledge of agent.
        """
        # save weights of local network
        torch.save(self.local_q_network.state_dict(), os.path.join(dir, 'checkpoint.pth'))

    def load(self, dir="./output"):
        """
        Load agent knowledge.

        Args:
            dir (str): Directory where knowledge (weigh files) is stored.
        """
        self.local_q_network.load_state_dict(torch.load(os.path.join(dir, 'checkpoint.pth')))

    def test(self):
        """
        Play a round with epsilon set to zero.

        Returns:
            score (int): achieved score
        """
        self.agent.eps = 0.0
        score = self.agent.play()
        return score
        