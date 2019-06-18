from base import FeaturesModel, Mark, GameState
import numpy as np
from typing import List, Tuple, Dict, Union, NamedTuple, Any
import logging

class Feature(NamedTuple):
    mark: int # my_mark=Mark.X.value, enemy_mark=Mark.O.value
    row: int
    enemy_behind: bool

class NumRowsFeatures(FeaturesModel):
    def __init__(self, row=5):
        self.row = row
        self._feature_names: List[Union[Feature, str]] = []
        for m in [Mark.X.value, Mark.O.value]:
            for b in [False, True]:
                for i in range(2, self.row + 1):
                    if i == self.row and b == True:
                        continue
                    self._feature_names.append(Feature(m, i, b))
        self._feature_names.append('alone_count')
        self._feature_names.append('alone_alone')
        # for example, current player is X, and features will be count of each:
        # XXXXX, XXXX , OXXXX , XXX  , OXXX  , XX   , OXX   ,
        # OOOO , XOOOO , OOO  , XOOO  , OO   , XOO   .

    def features_count(self) -> int:
        return ((((self.row - 1) * 2) - 1) * 2) + 2

    def feature_names(self) -> List[Any]:
        return self._feature_names

    def get_features(self, state: GameState)  -> List[int]:
        #logging.info(f'{self.__class__.__name__}.get_features()')
        board = state.board
        current_player = state.current_player
        occupied = np.where(board != Mark.NO.value)
        o = [(r, c) for r, c in zip(occupied[0], occupied[1])]

        features = dict((k, 0) for k in self._feature_names)
        #logging.info(len(features))

        if len(o) == 1:
            features['alone_count'] = 1
            features['alone_alone'] = 1
        else:
            features = self.find_rows(board, o, current_player)
            features['alone_count'] = self.get_alone(o)
            if sum([abs(f) for f in features.values()]) == 0:
                features['alone_alone'] = self.is_alone(o)
        #logging.info(len(features))
        return np.array(list(features.values()))

    def find_rows(self, board: np.ndarray, occupied: List[Tuple[int, int]], current_player: Mark) -> Dict[Union[Feature, str], int]:
        directions = [(0, 1), (1, 1), (1, 0), (1, -1)]
        h, w = board.shape

        features = dict((k, 0) for k in self._feature_names)

        v = np.zeros((h, w, len(directions)))

        no_value = Mark.NO.value
        curr_value = current_player.value

        for r, c in occupied:
            #print(f'r, c, val: {r}, {c}, {b[r, c]}')
            for d_i, (d_r, d_c) in enumerate(directions):
                #print(f'direction: {(d_r, d_c, d_i)}')
                if v[r, c, d_i] == 1: # we already count this cell in this direction
                    #print('already count')
                    continue
                m = board[r, c] # memorize mark
                #row = []
                #row.append((r, c))
                s = 1
                cnt = 1
                enemy_behind = False
                
                # look forward
                nr = r
                nc = c
                steal = True
                for _ in range(self.row):
                    nr += d_r
                    nc += d_c
                    if nr < 0 or nc < 0 or nr == h or nc == w: # beware of borders
                        if steal:
                            enemy_behind = True
                        break
                        
                    #if v[nr, nc, d_i] == 1: # perhaps, it is impossible
                    #    break
                    
                    nm = board[nr, nc]
                    #print(f'cell: {(nr, nc)}, {nm}')
                    #row.append((nr, nc))
                    if nm == m: # same mark
                        cnt += 1
                        v[nr, nc, d_i] = 1
                        steal = True
                    elif nm != no_value: # another mark
                        if steal:
                            enemy_behind = True
                        break
                    else:
                        steal = False # empty cell
                    s += 1
                    if s == self.row:
                        break
                
                #look backward:
                nr = r
                nc = c
                for i in range(self.row):
                    nr -= d_r
                    nc -= d_c
                    if nr < 0 or nc < 0 or nr == h or nc == w: # beware of borders
                        if steal:
                            enemy_behind = True
                        break
                        
                    nm = board[nr, nc]
                    #row.append((nr, nc))
                    if nm == m: # same mark, it can't be
                        cnt += 1
                        v[nr, nc, d_i] = 1
                    elif nm != no_value: # another mark
                        if i == 0:
                            enemy_behind = True
                        break
                    s += 1
                    if s >= self.row:
                        break
                    
                if s < self.row: # between enemies
                    continue
                
                if cnt > 1:
                    if cnt == self.row:
                        enemy_behind = False
                    if m == curr_value:
                        #rows.append((row, (1, cnt, enemy_behind)))
                        features[Feature(1, cnt, enemy_behind)] += 1
                    else:
                        #rows.append((row, (2, cnt, enemy_behind)))
                        features[Feature(2, cnt, enemy_behind)] += 1
        return features

    def get_alone(self, o: List[Tuple[int, int]]):
        alone_cnt = 0
        for p1 in o:
            np1 = np.array(p1)
            alone = True
            for p2 in o:
                if p1 == p2:
                    continue
                np2 = np.array(p2)
                if np.linalg.norm(np1 - np2) < 2:
                    alone = False
                    break
            if alone:
                alone_cnt += 1    
        return alone_cnt

    def is_alone(self, o: List[Tuple[int, int]]):
        for p1 in o:
            np1 = np.array(p1)
            for p2 in o:
                if p1 == p2:
                    continue
                np2 = np.array(p2)
                if np.linalg.norm(np1 - np2) < 2:
                    return 0
        return 1
