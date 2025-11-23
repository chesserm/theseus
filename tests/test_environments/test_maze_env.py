import pytest 
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src.config import *
from src.environments.maze_environment import MazeEnv 


def test_basic_config():
    try:
        env = gym.make(GYM_ENV_NAME)
        check_env(env.unwrapped)
    except Exception as e:
        assert False, f"{str(e)}"