import gymnasium as gym 
import numpy as np

from src.environments.maze_environment import Actions
from queue import Queue


class BFSAgent:
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
        self.bfs_queue = Queue()
        self.current_move_dir_queue = Queue()

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
        
        # Still performing move operations to handle the last position
        if (self.current_move_dir_queue.qsize() > 0):
            return self.current_move_dir_stack.get()


        # We've reached a new point. Add next possible options to stack
        # NOTE: We need to append 
        if (y + 1 < self.max_y and (x, y+1) not in self.visited):
            self.bfs_queue.put((x, y+1))
            self.bfs_queue.put((x,y))
        
        if (x + 1 < self.max_x and (x+1, y) not in self.visited):
            self.bfs_queue.put((x+1,y))
            self.bfs_queue.put((x,y))
        
        if (y - 1 >= 0 and (x, y - 1) not in self.visited):
            self.bfs_queue.put((x, y-1))
            self.bfs_queue.put((x,y))
        
        if (x - 1  >= 0 and (x - 1, y) not in self.visited):
            self.bfs_queue.put((x-1, y))
            self.bfs_queue.put((x,y))

        # Get the next position to travel to
        next_pos = self.bfs_queue.get()
        self.set_directions_to_next_pos(current_x=x, current_y=y, next_x=next_pos[0], next_y=next_pos[1])


    def set_directions_to_next_pos(self, current_x, current_y, next_x, next_y):

        num_ups = max(0, next_y - current_y)
        num_rights = max(0, next_x - current_x) 
        num_downs = max(0, current_y - next_y)
        num_lefts = max(0, current_x - next_x)

        for _ in range(num_ups):
            self.current_move_dir_queue.put(Actions.UP)

        for _ in range(num_rights):
            self.current_move_dir_queue.put(Actions.RIGHT)

        for _ in range(num_downs):
            self.current_move_dir_queue.put(Actions.DOWN)

        for _ in range(num_lefts):
            self.current_move_dir_queue.put(Actions.LEFT)
        
        return


    def get_path(self) -> list:
        return self.curr_path