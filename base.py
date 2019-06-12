from __future__ import annotations
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Any, Optional, Callable, NamedTuple

class Mark(Enum):
    NO = 0
    X = 1
    O = 2

class Action(NamedTuple):
    row: int
    col: int

@dataclass
class GameState:
    board: np.ndarray
    current_player: Mark
    is_game_end: bool
    winner: Mark

    def __str__(self) -> str:
        s = f'current_player: {self.current_player.name}\n'
        s += f'game ended: {self.is_game_end}\n'
        s += f'winner: {self.winner}\n'
        s += f'board:\n{str(self.board)}'
        return s

class View(ABC):
    @abstractmethod
    def __init__(self):
        self.turn_callback: Optional[Callable[[Action], None]] = None
    
    @abstractmethod
    def update(self, state: GameState) -> None:
        pass

class Agent(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_action(self, state: GameState) -> Action:
        pass
    
    @abstractmethod
    def win(self, state: GameState) -> None:
        pass
    
    @abstractmethod
    def loss(self, state: GameState) -> None:
        pass

    @abstractmethod
    def draw(self, state: GameState) -> None:
        pass

@dataclass
class Player:
    mark: Mark
    agent: Agent

class Policy(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_action(self, state: GameState, actions: Any):
        pass


