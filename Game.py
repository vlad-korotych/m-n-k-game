from base import Agent, View, Mark, Player, GameState, Action
from typing import Optional, IO
import numpy as np
import logging


class Game:
    def __init__(self,
                 player1: Agent,
                 player2: Agent,
                 m: Optional[int] = None,
                 n: Optional[int] = None,
                 k: int = 5,
                 view: Optional[View] = None,
                 play: bool = True,
                 debug: bool = False,
                 game_no: int = 0,
                 log_file: Optional[IO] = None
                 ):
        self.width: Optional[int] = m
        self.height: Optional[int] = n
        self.row: int = k # win row
        self.debug = debug
        self.log_file = log_file
        self.game_no = game_no

        width = m
        height = n

        auto_first_turn = False
        if self.width is None and self.height is None:
            # infinte board
            # let first turn will be on center of the board
            # and accessible field will be doubled row
            auto_first_turn = True

        if self.width is None:
            width = self.row * 2 + 1

        if self.height is None:
            height = self.row * 2 + 1

        self.board = np.full((height, width), Mark.NO.value)

        self.player1 = Player(Mark.X, player1)
        self.player2 = Player(Mark.O, player2)

        if auto_first_turn:
            self.board[self.row, self.row] = self.player1.mark.value
            self.current_player = self.player2
        else:
            self.current_player = self.player1

        self.view = view
        if self.view is not None:
            self.view.turn_callback = self.apply_action

        if self.debug:
            info = '\nGame.__init__'
            info += f'\nwidth: {self.width}'
            info += f'\nheight: {self.height}'
            info += f'\nrow: {self.row}'
            info += f'\nplayer1({self.player1.mark.name}): {self.player1.agent.__class__.__name__}'
            info += f'\nplayer2({self.player2.mark.name}): {self.player2.agent.__class__.__name__}'
            logging.info(info)

        if play:
            self.start()

    def apply_action(self, action: Action) -> None:
        row, column = action
        if self.log_file is not None:
            self.log_file.write(f'"{row}";"{column}"\n')
        if self.board[row][column] != Mark.NO.value:
            raise RuntimeError('Cell already used')
        self.board[row][column] = self.current_player.mark.value

        # check for win
        self.winner = self.check_win()

        if self.winner != Mark.NO:
            self.end_game = True
            # obviously, the winner is current player
            self.winner = self.current_player.mark

            # distribution of elephants
            if self.current_player == self.player1:
                self.player1.agent.win(self.new_state())
                self.player2.agent.loss(self.new_state())
            elif self.current_player == self.player2:
                self.player1.agent.loss(self.new_state())
                self.player2.agent.win(self.new_state())

        # check for draw
        elif len(self.board[self.board == Mark.NO.value]) == 0:
            self.end_game = True
            self.player1.agent.draw(self.new_state())
            self.player2.agent.draw(self.new_state())
        elif self.turn_no == 200:
            self.end_game = True
            self.player1.agent.draw(self.new_state())
            self.player2.agent.draw(self.new_state())
        else:
            # extend board
            if self.width is None or self.height is None:
                self.extend_board()

            if self.current_player == self.player1:
                self.current_player = self.player2
            elif self.current_player == self.player2:
                self.current_player = self.player1
            else:
                raise RuntimeError('Unknown player')

        self.state = self.new_state()
        self.update_view()

        if not self.end_game:
            self.turn_no += 1
            self.get_action()

    def get_action(self):
        if self.log_file is not None:
            h, w = self.state.board.shape
            self.log_file.write(f'"{self.game_no}";"{self.turn_no}";"{self.current_player.mark.name}";"{h}";"{w}";"{self.current_player.agent.get_learn_params()}";')
        action = self.current_player.agent.get_action(self.state)
        if action is None:
            return # wait callback from UI
        else:
            #row, column = action
            self.apply_action(action)

    def start(self) -> None:
        self.winner = Mark.NO
        self.end_game = False
        self.turn_no = 0

        self.state = self.new_state()
        if self.debug:
            logging.debug(f'\nInitial state:\n{self.state}')
        self.update_view()

        self.get_action()

    def new_state(self) -> GameState:
        state = GameState(self.board.copy(), self.current_player.mark, self.end_game, self.winner)
        return state

    def update_view(self):
        if self.view is not None:
            self.view.update(self.state)

    def check_win(self):
        h, w = self.board.shape

        # check rows
        for r in range(h):
            player = Mark.NO.value
            count = 0
            for c in range(w):
                player, count = self.check_cell(self.board[r, c], player, count)
                if count == self.row:
                    return player

        # check columns
        for c in range(w):
            player = Mark.NO.value
            count = 0
            for r in range(h):
                player, count = self.check_cell(self.board[r, c], player, count)
                if count == self.row:
                    return player

        #check diagonal left to right
        #top of the board
        for cs in range(w - (self.row - 1)):
            c = cs
            player = Mark.NO.value
            count = 0
            r = 0
            while True:
                player, count = self.check_cell(self.board[r, c], player, count)
                if count == self.row:
                    return player
                r += 1
                c += 1
                if r == h or c == w:
                    break

        # bottom of the board
        for rs in range(1, h - (self.row - 1)):
            r = rs
            player = Mark.NO.value
            count = 0
            c = 0
            while True:
                player, count = self.check_cell(self.board[r, c], player, count)
                if count == self.row:
                    return player
                r += 1
                c += 1
                if r == h or c == w:
                    break

        #check diagonal right to left
        #top of the board
        for cs in range(self.row - 1, w):
            c = cs
            player = Mark.NO.value
            count = 0
            r = 0
            while True:
                player, count = self.check_cell(self.board[r, c], player, count)
                if count == self.row:
                    return player
                c -= 1
                r += 1
                if c < 0 or r == h:
                    break

        # bottom of the board
        for rs in range(1, h - self.row + 1):
            r = rs
            player = Mark.NO.value
            count = 0
            c = w - 1
            while True:
                player, count = self.check_cell(self.board[r, c], player, count)
                if count == self.row:
                    return player
                c -= 1
                r += 1
                if c < 0 or r == h:
                    break

        return Mark.NO

    def check_cell(self, cell, player, count):
        if cell == Mark.NO.value:
            player = Mark.NO.value
            count = 0
        elif cell == player:
            count += 1
        else:
            player = cell
            count = 1

        return (player, count)

    def extend_board(self):
        top_rows = 0
        for i, r in enumerate(self.board):
            if r.sum() > 0:
                top_rows = i
                break

        bottom_rows = 0
        for i, r in enumerate(reversed(self.board)):
            if r.sum() > 0:
                bottom_rows = i
                break

        left_cols = 0
        for i, c in enumerate(self.board.T):
            if c.sum() > 0:
                left_cols = i
                break

        right_cols = 0
        for i, c in enumerate(reversed(self.board.T)):
            if c.sum() > 0:
                right_cols = i
                break

        add_top = self.row - top_rows
        add_bottom = self.row - bottom_rows
        add_left = self.row - left_cols
        add_right = self.row - right_cols

        top = np.zeros((add_top, self.board.shape[1]))
        bottom = np.zeros((add_bottom, self.board.shape[1]))

        left = np.zeros((self.board.shape[0] + add_top + add_bottom, add_left))
        right = np.zeros((self.board.shape[0] + add_top + add_bottom, add_right))

        self.board = np.c_[left, np.r_[top, self.board, bottom], right]
