from base import Chooser, GameState
import random
import numpy as np
from typing import List, Tuple, Any

class EpsilonGreedyChooser(Chooser):
    def __init__(self, epsilon: float, seed: int = None):
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)
            self.random_state = random.getstate()
        
        self.epsilon = epsilon
    
    def choose_action(self, state: GameState, actions: List[Tuple[Any, float]]) -> Any:
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
            max_weight = np.max([weight for action, weight in actions])
            action = random.choice([(action, weight) for action, weight in actions if weight == max_weight])
        
        return action