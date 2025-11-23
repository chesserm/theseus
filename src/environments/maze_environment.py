from enum import Enum
import gymnasium as gym
import numpy as np
from src.config import GYM_ENV_NAME, MAX_STEPS

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class MazeEnv(gym.Env):

    def __init__(self, shape=(10,10), render_mode=None):
        """
        Constructor for custom Maze Environment

        Heavily inspired by the example [GridWorldEnv shown in the gymnasium docs](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

        Inputs:
            - shape: An int or tuple (max_x, max_y) to determine grid size. Int creates a square of provided size
            - render_mode: The render_mode to use 
        """
        super().__init__()

        # TODO: Replace simple initialization with proper init from maze class
        if (type(shape) == int):
            self.grid = np.array((shape, shape), dtype=np.uint8)
            self.max_x = shape 
            self.max_y = shape
        elif (type(shape) == tuple and len(shape) == 2):
            self.grid = np.array(shape, dtype=np.uint8)
            self.max_x, self.max_y = shape
        else:
            raise RuntimeError(f"Invalid shape value provided: {shape}. Must be tuple specifying (rows, cols) or int for square.")

        self.observation_space = gym.spaces.Dict(
            {
                "agent" : gym.spaces.Box(low=np.array([0,0]), high=np.array([self.max_x, self.max_y]), shape=(2,), dtype=int),
                "target" : gym.spaces.Box(low=np.array([0,0]), high=np.array([self.max_x, self.max_y]), shape=(2,), dtype=int)
            }
        )

        # Use -1,-1 as an uninitialized state (reset hasn't been called)
        self._agent_location = np.array([-1, -1], dtype=np.int64)
        self._target_location = np.array([-1, -1], dtype=np.int64)

        # Up, Right, Down, Left (see Actions enum)
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0 : np.array([0, 1]),
            1 : np.array([1, 0]),
            2 : np.array([0, -1]),
            3 : np.array([-1, 0])
        }

        return 


    def _get_obs(self):
        """
        Converts internal state to observation format.

        Inputs:
            - None
        
        Outputs:
            - dict: Observation with agent and target positions
        """
        return {"agent" : self._agent_location, "target" : self._target_location}
    

    def _get_info(self):
        """
        Compute any auxilary information we want for debugging

        Inputs:
            - None
        
        Outputs:
            - dict: Info on distance between agent and target
        """
        # L1 distance
        return {
            "distance" : np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }
        

    def reset(self, seed=None, options=None):
        """
        Resets the state of the target and agent on the same sized grid.

        Inputs:
            - seed: Random seed for reproducible episodes
            - options: Additional configuration (unused in this example)
        
        Outputs:
            - tuple: (observation, info) for the initial state
        """

        # SUPER IMPORTANT: Reset RNG seed
        super().reset(seed=seed)

        # IMPORTANT: Several nuances must be handled here. 
        # 1. Use the parent class member variable for np_random, not global np.random for seed alignment
        # 2. Ensure the values being set to agent location are new arrays (not updating values in place)
        self._agent_location = np.array([
            self.np_random.integers(0, self.max_x),
            self.np_random.integers(0, self.max_y)
        ], dtype=int)

        self._target_location = self._agent_location.copy()
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location[0] = self.np_random.integers(0, self.max_x)
            self._target_location[1] = self.np_random.integers(0, self.max_y)
        
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info)


    def step(self, action):
        """
        The core update function. 

        Inputs:
            - action: The action to take. Must be one of the Actions enum possibilities
        
        Outputs:
            - observation: The current observation after the action
            - reward: The reward obtained from the action
            - target_reached: True if target was found
            - truncated: Overwritten by TimeLimit wrapper. Left as False here.
            - info: The info after the action is taken
        """
        
        # Convert from action to direction (np array)
        np_direction = self._action_to_direction[action]

        # Move agent according to direction - ensuring new pos. is on grid
        self._agent_location = self._agent_location + np_direction
        self._agent_location[0] = np.clip(self._agent_location[0], 0, self.max_x - 1)
        self._agent_location[1] = np.clip(self._agent_location[1], 0, self.max_y - 1)

        # See if target has been reached
        target_reached = np.array_equal(self._agent_location, self._target_location)

        # Define reward
        # TODO: Improve
        reward = 1 if target_reached else -0.01

        observation = self._get_obs()
        info = self._get_info()

        # NOTE: This is overwritten by the TimeLimit wrapper when max steps is exceeded. Set to false by default here.
        truncated = False

        return observation, reward, target_reached, truncated, info

    
    def render(self):
        """
        Render the Environment for Human Viewing and Debugging.
        """

        if (self.render_mode == "human"):
            for j in range(self.max_y - 1, -1, -1):
                row = ""
                for i in range(self.max_x):
                    if (np.array_equal([i,j], self._agent_location)):
                        row += "A " # For agent
                    elif (np.array_equal([i,j], self._target_location)):
                        row += "T " # For target
                    else:
                        row += ". "
                print(row)
            print()
        return


gym.register(
    id=GYM_ENV_NAME,
    entry_point=MazeEnv,
    max_episode_steps=MAX_STEPS
)

