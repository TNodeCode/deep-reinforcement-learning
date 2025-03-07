import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from src.agent.simple import SimpleAgent
from src.network.actor import Actor
from src.network.critic import Critic
from src.network.utils import NetworkUtils


class DDPGAgent(SimpleAgent):
    def __init__(
            self,
            env,
            eps=1.0,
            lr_actor=1e-4,
            lr_critic=3e-4,
            hidden_dims_actor=[64,64],
            hidden_dims_critic=[64,64],
            tau=1e-2,
            memory_size=int(1e6),
            max_steps=1_000,
            batch_size=64,
            device="cpu"
    ):
        """
        Constructor.
        """
        super().__init__(env, eps=eps, memory_size=memory_size, batch_size=batch_size, max_steps=max_steps, action_space_discrete=False, device=device)
        self.eps = eps
        self.gamma = torch.tensor(0.99).to(device)
        self.tau = tau
        self.update_every = 1
        self.device = device

        # Build a local and a target actor network
        self.actor_local = Actor(
            dim_state=self.env.get_state_shape()[-1],
            dim_action=self.env.get_action_shape()[-1],
            hidden_dims=hidden_dims_actor,
        ).to(device)        
        self.actor_target = Actor(
            dim_state=self.env.get_state_shape()[-1],
            dim_action=self.env.get_action_shape()[-1],
            hidden_dims=hidden_dims_actor,
        ).to(device)

        # Build a local and a target critic network
        self.critic_local = Critic(
            dim_state=self.env.get_state_shape()[-1],
            dim_action=self.env.get_action_shape()[-1],
            hidden_dims=hidden_dims_critic,
        ).to(device)        
        self.critic_target = Critic(
            dim_state=self.env.get_state_shape()[-1],
            dim_action=self.env.get_action_shape()[-1],
            hidden_dims=hidden_dims_critic,
        ).to(device)

        # Copy weights from local actor and local critic to target networks
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # optimizers
        self.optim_actor = torch.optim.AdamW(self.actor_local.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.AdamW(self.critic_local.parameters(), lr=lr_critic)

    def choose_action(self):
        """
        Choose an action based on internal knowledge.

        Returns:
            torch.tensor: best action as continuous vector
        """
        # get a continuous representation of the optimal action
        self.actor_local.eval()
        state_tensor = torch.from_numpy(self.state).float().to(self.device)
        action = self.actor_local(state_tensor)
        # add random noise to action tensor, this will help the model to stabilize predictions
        #action += 0.005 * torch.randn(*action.shape)
        self.actor_local.train()
        action = np.clip(action.detach().cpu().numpy(), -1, 1)
        return action

    def learn(self):
        """
        Learning step.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # get a random batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size=self.batch_size)

        ## --- Critic update --- ##

        # actor computes action vector based on state
        target_actions = self.actor_target(states=next_states)
        target_q_values = self.critic_target(states=next_states, actions=target_actions)
        target_values = rewards + self.gamma * target_q_values * (1 - dones)

        # compute difference between expected / target values that local / target critic computed
        critic_values = self.critic_local(states=states, actions=actions)
        loss_critic = F.mse_loss(critic_values, target_values)
        
        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

        ## --- Actor update --- ##

        loss_actor = -self.critic_local(states=states, actions=self.actor_local(states=states)).mean()
        
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        ## --- Target network updates --- ##

        NetworkUtils.soft_update(local_model=self.actor_local, target_model=self.actor_target, tau=self.tau)
        NetworkUtils.soft_update(local_model=self.critic_local, target_model=self.critic_target, tau=self.tau)