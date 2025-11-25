import gymnasium as gym 
import numpy as np 

class TabularQLearningAgent:
    def __init__(
        self, 
        env: gym.Env,
        learning_rate:float,
        initial_epsilon:float,
        epsilon_decay:float,
        final_epsilon:float,
        discount_factor:float=0.95
    ):
        """
        Constructor for Q-Learning Agent.

        Heavily inspired by the [gym docs example on Blackjack](https://gymnasium.farama.org/introduction/train_agent/)

        Inputs:
            - gym: The gymnasium environment
            - learning_rate: Rate for updating Q-values (0, 1)
            - initial_epsilon: Starting exploration rate. Higher = more exploration than exploitation
            - epsilon_decay: How much to reduce epsilon per episode
            - final_epsilon: Minimum exploration rate (0, 1)
            - discount_factor: How much to value future rewards
        """
        # Store some important values about the environment
        self.env = env
        self.max_x = self.env.max_x
        self.max_y = self.env.max_y
        self.num_actions = len(self.env.action_space)

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Create Q-Table
        # NOTE: lazy initialization here. See how this is handled in __get_q_table_value() and __set_q_table_value()
        self.q_table = {}

        # For tracking
        self.training_error = []
        return 
    
    
    def __get_q_table_value(self, obs, action=None):
        """
        Returns the value the obs in the Q-table. 

        This is more efficient than prepopulating a massive table.

        Input:
            - obs: The observation
            - action: The action selection
        
        Output:
            - The Q-table value.
        """

        if (obs not in self.q_table):
            self.q_table[obs] = np.zeros((self.num_actions,))
        
        if (action is not None):
            return self.q_table[obs][action]
        
        return self.q_table[obs]


    def __set_q_table_value(self, obs, action, value):
        """
        Sets the value the obs in the q-table. 

        This is more efficient than prepopulating a massive table.

        Input:
            - obs: The observation
            - action: The action selection
            - value: The value to set
        """

        if (obs not in self.q_table):
            self.q_table[obs] = np.zeros((self.num_actions,))
        
        self.q_table[obs][action] = value
        return 


    def get_action(self, obs) -> int:
        """
        Choose an action based on observation

        Inputs:
            - obs: The observation from this point in the search
        
        Outputs:
            - action: An selection for the next action from the set of possible options
        """

        # If random value < epsilon, then explore (random option), otherwise exploit
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        
        return int(np.argmax(self.__get_q_table_value(obs)))
    

    def update(self, obs, action, reward, terminated, next_obs) -> None:
        """
        Update Q-table based on observed behavior
        """

        future_q_value = (0 if terminated else 1) * np.max(self.__get_q_table_value(next_obs))

        # Bellman Equation for Q value
        target = reward + (self.discount_factor * future_q_value)

        # Loss (how wrong our current estimate of target was) 
        loss = target - self.__get_q_table_value(obs, action=action)
        self.training_error.append(loss)

        # Q table update
        new_value = self.__get_q_table_value(obs, action=action) + self.lr * loss
        self.__set_q_table_value(obs, action, new_value)

        return 
    

    def decay_eps(self) -> None:
        """
        Updates epsilon according to strategy
        """

        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        return
