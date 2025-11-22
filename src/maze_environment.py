from enum import Enum
import gymnasium as gym
import numpy as np
import pygame
from maze import Maze


class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class MazeEnv(gym.Env):
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps" : 4}

    def __init__(self, shape=(10,10), render_mode=None):
        """
        Constructor for custom Maze Environment

        Heavily inspired by the example [GridWorldEnv shown in the gymnasium docs](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

        Inputs:
            - shape: An int or tuple (m,n) to determine grid size. Int creates a square of provided size
            - render_mode: The render_mode to use 
        """
        super().__init__()

        # TODO: Replace simple initialization with proper init from maze class
        if (type(shape) == int):
            self.grid = np.array((shape, shape), dtype=np.uint8)
            self.num_rows = shape 
            self.num_cols = shape
        elif (type(shape) == tuple and len(shape) == 2):
            self.grid = np.array(shape, dtype=np.uint8)
            self.num_rows, self.num_cols = shape
        else:
            raise RuntimeError(f"Invalid shape value provided: {shape}. Must be tuple specifying (rows, cols) or int for square.")

        # The size of the frame for creating animations with pygame 
        self.window_size = 512

        self.observation_space = gym.spaces.Dict(
            {
                "agent" : gym.spaces.Box(low=np.array([0,0]), high=np.array([self.num_cols, self.num_rows]), shape=(2,), dtype=int),
                "target" : gym.spaces.Box(low=np.array([0,0]), high=np.array([self.num_cols, self.num_rows]), shape=(2,), dtype=int)
            }
        )

        self._agent_location = np.array([-1, -1], dtype=np.uint8)
        self._target_location = np.array([-1, -1], dtype=np.uint8)


        # Up, Right, Down, Left (see Actions enum)
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            Actions.UP : np.array([0, 1]),
            Actions.RIGHT : np.array([1, 0]),
            Actions.DOWN : np.array([0, -1]),
            Actions.LEFT : np.array([-1, 0])
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

        Inputs:
            - seed: Random seed for reproducible episodes
            - options: Additional configuration (unused in this example)
        
        Outputs:
            - tuple: (observation, info) for the initial state
        """

        # SUPER IMPORTANT: Reset RNG seed
        super().reset(seed=seed)

        # TODO: Change from this super simple approach to randomly initialize grid, target location, and agent location
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info)


    def step(self, action):
        """

        """
        # TODO: Implement  https://gymnasium.farama.org/introduction/create_custom_env/#step-function
        raise NotImplementedError()

