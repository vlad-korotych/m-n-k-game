from base import Agent, GameState, Mark
from typing import Tuple

class ConsoleHumanAgent(Agent):
    def __init__(self):
        pass
    
    def get_action(self, state: GameState) -> Tuple[int, int]:
        h, w = state.board.shape
        while True:
            try:
                inp = tuple(int(x) for x in input('Please, make your turn: row,column: ').split(','))
                r, c = int(inp[0]), int(inp[1])
            except ValueError:
                print('\nInvalid input, try again')
                continue
            if r not in range(h):
                print('\nRow is out of range, try again')
                continue
            if c not in range(w):
                print('\nColumn is out of range, try again')
                continue
            if state.board[r, c] != Mark.NO.value:
                print(f'\nRow {r}, column {c} is occupied, try again')
                continue
            break
        return r, c

    def win(self, state: GameState):
        pass

    def loss(self, state: GameState):
        pass

    def draw(self, state: GameState):
        pass