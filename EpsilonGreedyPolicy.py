from base import Policy, GameState, Action
import random
import numpy as np
from typing import List, Tuple, Any
from QLearningApproximationAgent import ActionInfo

class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float, seed: int = None):
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)
            self.random_state = random.getstate()
        
        self.epsilon = epsilon
    
    def get_action(self, state: GameState, actions: List[ActionInfo]) -> ActionInfo:
        """
        actions is list of (action, weight),
        """
            
        if self.epsilon > random.random():
            # exploration
            if self.seed is not None:
                random.setstate(self.random_state)
                
            action = random.choice(actions)
            
            if self.seed is not None:
                self.random_state = random.getstate()
        else:
            # exploitation
            max_Q_value = np.max([a.new_Q_value for a in actions])
            action = random.choice([a for a in actions if a.new_Q_value == max_Q_value])
       
        return action