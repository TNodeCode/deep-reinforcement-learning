import numpy as np
from src.memory.memory import Memory
from src.environment.abstract import AbstractEnvironment
from tqdm import tqdm


class SimpleAgent:
    """Simple Agent is an agent that does not use any learning algorithm. It interacts randomly with the environment.
    """
    def __init__(
            self,
            env: AbstractEnvironment,
            memory_size=int(1e5),
            batch_size=256,
            action_space_discrete=True,
            max_steps=100,
            device="cpu",
            **kwargs
        ):
        """
        Constructor.
        
        Params:
            env: Abstract^ environment
        """
        self.env = env
        self.max_steps = max_steps
        self.memory = Memory(maxlen=memory_size, action_space_discrete=action_space_discrete, device=device) if memory_size else None
        self.batch_size = batch_size
        self.state = None
        self.score = 0
        self.current_step = 0
        self.agent_params = kwargs
        self.reset()
        
    def reset(self):
        """
        Reset the environment.
        """
        self.env.reset()
        self.state = self.env.get_current_state()
        self.score = 0
        self.current_step = 0
        
    def update_state(self):
        """
        Get current state from environment and update internal state values.
        
        Returns:
            (int, int): tuple containing state before and after update
        """
        previous_state = self.state
        next_state = self.env.get_current_state()
        self.state = next_state
        return previous_state, next_state
        
    def update_memory(self, states, actions, rewards, next_states, dones):
        """
        Save step results in memory.
        
        Args:
            states: states before step was taken
            actions: actions that was taken by the step
            rewards: rewards that was earned by performing the step
            next_states: states after step was performed
            dones: indocator whether game was over after step
        """
        self.memory.append(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones
        )
    
    def choose_action(self):
        """
        Choose an action.
        
        Returns:
            int: Action
        """
        return np.array([[np.random.randint(self.env.get_num_actions())]])
    
    def do_step(self, actions):
        """
        Perform a step based on a chosen action.

        Args:
            actions: (N_a x D_a) Array representing action for each agent
        
        Returns:
            step results (tuple): State before step, reward, state after step, indicator if agent has reached goal
        """
        # perform action
        rewards = self.env.step(actions)
        # get the next state
        states, next_states = self.update_state()
        # update score
        self.score += sum(rewards) / len(rewards)
        # check if game is over
        dones = self.env.is_done()
        return states, rewards, next_states, dones
    
    def learn(self):
        """
        Method for learning based on experience.
        """
        return # this is a simple agent, it wn't learn anything.
        
    def play(self):
        """
        Play the game until it is finished.
        
        Args:
            max_steps (int): Maximum number of steps in one epoch
        """
        self.current_step = 0
        for step in range(self.max_steps):
            self.current_step = step
            actions = self.choose_action()
            states, rewards, next_states, dones = self.do_step(actions=actions)
            if self.memory is not None:
                self.update_memory(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)
            self.learn()
            if all(dones):
                break
        return self.score
    
    def save(self):
        """
        Save status of agent.
        """
        return # As the simple agent doesn't learn anything there is nothing to save here.