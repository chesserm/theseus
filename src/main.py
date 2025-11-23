from src.utils.maze import Maze 
from src.environments.maze_environment import MazeEnv
from traceback import format_exc
import gymnasium as gym
import random
from gymnasium.utils.env_checker import check_env
import os 

from src.config import MAX_STEPS, GYM_ENV_NAME

def main():

    env = gym.make(GYM_ENV_NAME, max_episode_steps=MAX_STEPS)
    
    try:
        check_env(env.unwrapped)
        print("Environment passes all checks")
    except Exception as e:
        print(f"Error checking environment: {str(e)}. Traceback {format_exc()}")

    env.reset(seed=random.randint(0, 1000))

    episode_over = False
    total_reward = 0
    steps_taken = 0
    while (not episode_over):
        # Choose an action
        action = env.action_space.sample()

        # Take this action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)
        steps_taken += 1

        total_reward += reward
        episode_over = terminated or truncated or steps_taken == MAX_STEPS

    print("Episode Finished") 
    if (MAX_STEPS == steps_taken or truncated):
        print(f"Terminated due to max number of steps taken {steps_taken}. Did not find target")
    else:
        print(f"Congrats! The Agent found the target in {steps_taken} steps")
    

    return


if __name__ == "__main__":
    main()

