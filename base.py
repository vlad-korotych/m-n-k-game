from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Any

class Mark(Enum):
    NO = 0
    X = 1
    O = 2

@dataclass
class GameState:
    board: np.ndarray
    current_player: Mark
    is_game_end: bool
    winner: Mark

class View(ABC):
    @abstractmethod
    def __init__(self):
        self.turn_callback = None
    
    @abstractmethod
    def update(self, state: GameState):
        pass

class Agent(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_action(self, state: GameState):
        pass
    
    @abstractmethod
    def win(self, state: GameState):
        pass
    
    @abstractmethod
    def loss(self, state: GameState):
        pass

    @abstractmethod
    def draw(self, state: GameState):
        pass

@dataclass
class Player:
    mark: Mark
    agent: Agent

class Chooser(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def choose_action(self, state: GameState, actions: List[Tuple[Any, float]]):
        pass