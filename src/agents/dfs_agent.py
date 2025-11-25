import gymnasium as gym 
import numpy as np

from src.environments.maze_environment import Actions

class DFSAgent:
    def __init__(
        self, 
        env: gym.Env
    ):
        """
        Constructor for DFS Agent.

        Inputs:
            - gym: The gymnasium environment
        """
        # Store some important values about the environment
        self.env = env
        self.max_x = self.env.max_x
        self.max_y = self.env.max_y
        self.num_actions = len(self.env.action_space)

        self.curr_path = []
        self.visited = {}

        return 
    
    def get_action(self, obs) -> int:
        """
        Choose an action based on observation

        Inputs:
            - obs: The observation from this point in the search (Agent position)
        
        Outputs:
            - action: An selection for the next action from the set of possible options
        """
        self.curr_path.append(obs)
        x,y = obs
        self.visited[(x,y)] = True

        # DFS in clockwise order: Top, Right, Bottom, Left
        if (y + 1 < self.max_y and (x, y+1) not in self.visited):
            return Actions.UP 
        
        if (x + 1 < self.max_x and (x+1, y) not in self.visited):
            return Actions.RIGHT 
        
        if (y - 1 >= 0 and (x, y - 1) not in self.visited):
            return Actions.DOWN
        
        return Actions.LEFT
        

    def get_path(self) -> list:
        return self.curr_path