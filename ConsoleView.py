from base import Mark, View, GameState, Action
import math
from typing import Callable, Optional

class ConsoleView(View):
    def __init__(self) -> None:
        self.turn_callback: Optional[Callable[[Action], None]] = None

    def update(self, state: GameState) -> None:
        self.print_board(state)

    def print_board(self, state: GameState) -> None:

        label: str = '\n\n'
        if state.is_game_end:
            if state.winner == Mark.NO:
                label = "Draw!"
            else:
                label = state.winner.name + ' wins!'
        else:
            label = state.current_player.name + "'s turn"
        label += '\n'

        h, w = state.board.shape
        
        cell_w = self.num_width(w)
        rownum_w = self.num_width(h)

        board: str = ''
        board += self.row_of_num(w, cell_w, rownum_w)

        for r in range(h):
            board += self.line_row(w, cell_w, rownum_w)
            row = self.num_cell(r, rownum_w)
            for c in range(w):
                row += '|'
                if state.board[r, c] == Mark.NO.value:
                    row += ' ' * cell_w
                else:
                    row += Mark(state.board[r, c]).name
                    row += ' ' * (cell_w - 1)
            row += f'|{r}\n'
            board += row
        board += self.row_of_num(w, cell_w, rownum_w)
        
        print(label)
        print(board)

    def row_of_num(self, w: int, cell_w: int, rownum_w: int) -> str:
        row: str = ' ' * rownum_w
        for c in range(w):
            row += ' '
            row += self.num_cell(c, cell_w)
        row += '\n'
        return row
    
    def line_row(self, w, cell_w, rownum_w) -> str:
        s = '-' * (rownum_w + 1 + (cell_w + 1) * w)
        return s + '\n'

    def num_width(self, num) -> int:
        if num == 0:
            return 1
        return math.ceil(math.log10(num+1))

    def num_cell(self, num, cell_w) -> str:
        nw = self.num_width(num)
        s = f'{num}'
        s += ' ' * (cell_w - nw)
        return s



