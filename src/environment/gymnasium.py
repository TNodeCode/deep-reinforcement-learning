import gymnasium as gym
import ale_py
import numpy as np
from src.environment.abstract import AbstractEnvironment

gym.register_envs(ale_py)


class GymnasiumEnvironment(AbstractEnvironment):
    """
    The GymnasiumEnvironment class is a wrapper for interacting with environments from the Gymnasium library.
    It inherits from the AbstractEnvironment class and implements its abstract methods.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initializes the Gymnasium environment with the given environment name.

        Args:
            name (str): The name of the Gymnasium environment.
        """
        # Instantiate the Gymnasium environment
        if "render_mode" not in kwargs:
            kwargs.update({"render_mode": None})
        super().__init__(name=name, **kwargs)

    def can_render(self) -> bool:
        """
        Check if environment can be rendered.
        """
        return False

    def close(self) -> None:
        """
        Close the environment.
        """
        self.env.close()

    def get_action_shape(self) -> tuple[int]:
        """
        Returns the shape of the action space.

        Returns:
            int: The shape of the action space.
        """
        return self.action_shape

    def get_num_actions(self) -> int:
        """Get number of available actions.

        Returns:
            int: Number of available actions
        """
        if self.is_action_space_continuous():
            return np.inf
        return int(self.env.action_space.n)

    def get_num_agents(self) -> int:
        """
        Returns the number of agents in the environment. For Gymnasium, this is always 1.

        Returns:
            int: Number of agents.
        """
        return 1

    def get_reward_shape(self) -> tuple[int]:
        """
        Returns the shape of the reward space.

        Returns:
            int: The shape of the reward space.
        """
        return self.reward_shape

    def get_state_shape(self) -> tuple[int]:
        """
        Returns the shape of the state space.

        Returns:
            int: The shape of the state space.
        """
        return self.state_shape

    def init_env(self, name: str):
        """Initialize environment.

        Args:
            name (str): Name of the environment
        """
        self.env = gym.make(name, **self.env_params)
        self.state_space_continuous = isinstance(self.env.observation_space, gym.spaces.Box)
        self.action_space_continuous = isinstance(self.env.action_space, gym.spaces.Box)
        if self.can_render():
            self.env.reset()
            observation = self.render()
            self.env.close()
            self.state_shape = observation.shape
        else:
            self.state_shape = (self.get_num_agents(), self.env.observation_space.shape[0])
        self.action_shape = (self.get_num_agents(), int(self.env.action_space.shape[0]) if self.is_action_space_continuous() else 1)
        self.reward_shape = (self.get_num_agents(), 1)

    def is_action_space_continuous(self) -> bool:
        """
        Check if the action space is continuous.

        Returns:
            bool: True if the action space is continuous, False if discrete.
        """
        return self.action_space_continuous

    def is_state_space_continuous(self) -> bool:
        """
        Check if the state space is continuous.

        Returns:
            bool: True if the state space is continuous, False if discrete.
        """
        return self.state_space_continuous

    def is_done(self) -> np.ndarray:
        """
        Returns whether the current episode has ended.

        Returns:
            np.ndarray: Array indicating whether the episode is done.
        """
        return np.array([self.terminated])

    def render(self) -> np.ndarray:
        """
        Render the environment.
        """
        return self.env.render()

    def reset(self) -> None:
        """
        Resets the Gymnasium environment to its initial state.
        """
        self.env.reset()

    def step(self, actions: np.ndarray) -> np.ndarray:
        """
        Executes an action in the Gymnasium environment and returns the reward.
        See: https://gymnasium.farama.org/api/env/#gymnasium.Env.step

        Args:
            actions (np.ndarray): Action vector of shape (n_agents, action_dim) or (action_dim).

        Returns:
            np.ndarray: Reward for the action taken.
        """
        observation, reward, terminated, truncated, info = self.env.step(actions.squeeze(axis=0))
        self.current_state = np.expand_dims(observation, axis=0)
        self.last_action = actions
        self.last_reward = np.array([reward])
        self.terminated = terminated
        return np.array([reward])
    

class VisualGymnasiumEnvironment(GymnasiumEnvironment):
    """
    The GymnasiumEnvironment class is a wrapper for interacting with environments from the Gymnasium library.
    It inherits from the AbstractEnvironment class and implements its abstract methods.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initializes the Gymnasium environment with the given environment name.

        Args:
            name (str): The name of the Gymnasium environment.
        """
        # Instantiate the Gymnasium environment
        kwargs.update({"render_mode": "rgb_array"})
        super().__init__(name=name, **kwargs)

    def can_render(self) -> bool:
        """
        Check if environment can be rendered.
        """
        return True

    def init_env(self, name: str):
        """Initialize environment.

        Args:
            name (str): Name of the environment
        """
        super().init_env(name)
        self.env.reset()
        self.state_dim = self.env.render().shape
    
    def step(self, actions: np.ndarray) -> np.ndarray:
        """
        Executes an action in the Gymnasium environment and returns the reward.
        See: https://gymnasium.farama.org/api/env/#gymnasium.Env.step

        Args:
            actions (np.ndarray): Action vector of shape (action_dim,).

        Returns:
            np.ndarray: Reward for the action taken.
        """
        observation, reward, terminated, truncated, info = self.env.step(actions.squeeze())
        self.current_state = np.expand_dims(self.env.render(), axis=0)
        self.last_action = actions
        self.last_reward = np.array([reward])
        self.terminated = terminated
        return np.array([reward])
