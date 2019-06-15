from base import Policy, GameState, Action
import random
import numpy as np
from typing import List, Tuple, Any
from QLearningApproximationAgent import ActionInfo
import logging

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
        logging.info(f'{self.__class__.__name__}.get_action()')
        if self.epsilon > random.random():
            # exploration
            logging.info('exploration')
            if self.seed is not None:
                random.setstate(self.random_state)
                
            action = random.choice(actions)
            
            if self.seed is not None:
                self.random_state = random.getstate()
        else:
            # exploitation
            info = 'exploitation\n'
            max_Q_value = np.max([a.new_Q_value for a in actions])
            max_actions = [a for a in actions if a.new_Q_value == max_Q_value]
            info += f'max_Q_value: {max_Q_value}, total values: {len(actions)}, with max Q_value: {len(max_actions)}\n'
            logging.info(info)
            action = random.choice(max_actions)

        return action