import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvironment(ABC):
    """
    The AbstractEnvironment class is a wrapper for interacting with environments like Gymnasium or Unity ML Agent environments.
    It provides an interface to manage and interact with multiple agents in either continuous or discrete action and state spaces.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initializes the environment with the given name and space continuity properties.

        Args:
            name (str): The name of the environment.
        """
        self.env = None
        self.env_params = kwargs
        self.init_env(name)
        self.current_state = np.empty(self.get_state_shape())
        self.last_action = np.empty(self.get_action_shape())
        self.last_reward = np.empty(self.get_reward_shape())

    @abstractmethod
    def close(self) -> None:
        """
        Close the environment.
        """
        pass

    @abstractmethod
    def get_action_shape(self) -> tuple[int]:
        """
        Get the shape of actions that this environment accepts.

        Returns:
            int: The shape of the action space.
        """
        pass

    def get_current_state(self) -> np.ndarray:
        """
        Get the current state of the environment.

        Returns:
            np.ndarray: State represented by a vector of shape (number_of_agents, state_dim).
        """
        return self.current_state

    def get_last_action(self) -> np.ndarray:
        """
        Get the last action taken.

        Returns:
            np.ndarray: Reward represented by a float vector of shape (number_of_agents, 1).
        """
        return self.last_action

    def get_last_reward(self) -> np.ndarray:
        """
        Get the last reward received by the last action taken.

        Returns:
            np.ndarray: Reward represented by a float vector of shape (number_of_agents, 1).
        """
        return self.last_reward

    @abstractmethod
    def get_num_actions(self) -> int:
        """Get number of available actions.

        Returns:
            int: Number of available actions
        """
        pass

    @abstractmethod
    def get_num_agents(self) -> int:
        """
        Get the number of agents in this environment.

        Returns:
            int: The number of agents.
        """
        pass

    @abstractmethod
    def get_reward_shape(self) -> tuple[int]:
        """
        Get the shape of the reward of this environment.

        Returns:
            int: The shape of the reward space.
        """
        pass

    @abstractmethod
    def get_state_shape(self) -> tuple[int]:
        """
        Get the shape of the state of this environment.

        Returns:
            int: The shape of the state space.
        """
        pass

    @abstractmethod
    def is_action_space_continuous(self) -> bool:
        """
        Check if the action space is continuous.

        Returns:
            bool: True if the action space is continuous, False if discrete.
        """
        pass

    @abstractmethod
    def init_env(self, name: str):
        """
        Initialize the environment.

        Args:
            name (str): Name of the environment.
        """
        pass

    @abstractmethod
    def is_done(self) -> np.ndarray:
        """
        Check if agents can still act or if they reached their goal.

        Returns:
            np.ndarray: Binary integer array containing zeros and ones of shape (number_of_agents, 1).
        """
        pass

    @abstractmethod
    def is_state_space_continuous(self) -> bool:
        """
        Check if the state space is continuous.

        Returns:
            bool: True if the state space is continuous, False if discrete.
        """
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        """
        Render the environment.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the environment.
        """
        pass

    @abstractmethod
    def step(self, actions: np.ndarray) -> np.ndarray:
        """
        Each agent executes an action and receives a reward.

        Args:
            actions (np.ndarray): Action vector of shape (number_of_agents, action_dim).

        Returns:
            np.ndarray: Rewards for each agent represented by a float array of shape (number_of_agents, 1).
        """
        pass