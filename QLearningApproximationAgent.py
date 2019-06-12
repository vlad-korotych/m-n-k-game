from base import Agent, Policy, Mark, Action, GameState
import numpy as np
from typing import List, Optional, Dict, NamedTuple, Union, Tuple
import logging
import random
from dataclasses import dataclass

class Feature(NamedTuple):
    mark: int # my_mark=Mark.X.value, enemy_mark=Mark.O.value
    row: int
    enemy_behind: bool

@dataclass
class ActionInfo:
    action: Action
    new_board: Optional[np.ndarray] = None
    new_features: Optional[np.ndarray] = None
    new_Q_value: Optional[float] = None

class QLearningApproximationAgent(Agent):
    def __init__(self, alfa: float, gamma: float, policy: Policy, row: int = 5, learning: bool = True, theta: Optional[np.ndarray]=None, inf_field: bool = True):
        self.alfa: float = alfa
        self.gamma: float = gamma
        self.policy: Policy = policy
        self.is_learning: bool = learning
        self.row: int = row
        self.history_actions: List[ActionInfo] = []
        self.history_theta: List[np.ndarray] = []
        self.inf_field: bool = inf_field
        
        self.feature_names: List[Union[Feature, str]] = []
        for m in [Mark.X.value, Mark.O.value]:
            for b in [False, True]:
                for i in range(2, self.row + 1):
                    if i == self.row and b == True:
                        continue
                    self.feature_names.append(Feature(m, i, b))
        self.feature_names.append('alone')
        # for example, current player is X, and features will be count of each:
        # XXXXX, XXXX , OXXXX , XXX  , OXXX  , XX   , OXX   ,
        # OOOO , XOOOO , OOO  , XOOO  , OO   , XOO   .


        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.zeros(len(self.feature_names))

    
    def get_action(self, new_state: GameState) -> Action:
        logging.info(f'{self.__class__.__name__}.get_action()')
        actions = self.get_possible_actions(new_state)
        logging.info(f'Player: {new_state.current_player.name}')
        logging.info(f'\nPossible actions: {len(actions)}')

        actions = self.get_actions_features(new_state, actions)

        actions = self.Q_values(actions)

        Q_table = new_state.board.copy()
        Q_table[Q_table == Mark.X.value] = Mark.X.name
        Q_table[Q_table == Mark.O.value] = Mark.O.name
        for a in actions:
            info = f'\nAction: {a.action}\n'
            info += f'Board:\n{a.new_board}\n'
            info += f'Features: {a.new_features}\n'
            info += f'Q value: {a.new_Q_value}'
            Q_table[a.action.row, a.action.col] = a.new_Q_value
            logging.info(info)
        
        logging.info(f'Q table:\n{Q_table}')

        if len(self.history_actions) > 0 is not None and self.is_learning:
            reward = -1
            self.learn(self.history_actions[-1], actions, reward)
        
        action = self.policy.get_action(new_state, actions)

        logging.info(f'Action: {action.action}')
        
        self.history_actions.append(action)

        return action.action
    
    def get_actions_features(self, state: GameState, actions: List[ActionInfo]) -> List[ActionInfo]:
        for a in actions:
            a.new_board = state.board.copy()
            a.new_board[a.action.row, a.action.col] = state.current_player.value
            a.new_features = self.get_features(a.new_board, state.current_player)
        return actions
    
    def get_possible_actions(self, state: GameState) -> List[ActionInfo]:
        f = np.where(state.board == Mark.NO.value)
        possible_actions = [ActionInfo(action=Action(r, c)) for r, c in zip(f[0], f[1])]
        return possible_actions
    
    def get_features(self, board: np.ndarray, current_player: Mark) -> List[int]:

        no_value = Mark.NO.value
        curr_value = current_player.value

        directions = [(0, 1), (1, 1), (1, 0), (1, -1)]
        h, w = board.shape

        features = dict((k, 0) for k in self.feature_names)

        v = np.zeros((h, w, len(directions)))

        occupied = np.where(board != Mark.NO.value)
        o = [(r, c) for r, c in zip(occupied[0], occupied[1])]

        if len(o) == 1:
            features['alone'] = 1
        else:
            for r, c in o:
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
            if sum([abs(f) for f in features.values()]) == 0:
                features['alone'] = self.check_alone(o)
        #print(rows)
        return np.array(list(features.values()))
    
    def check_alone(self, o: List[Tuple[int, int]]):
        for p1 in o:
            np1 = np.array(p1)
            for p2 in o:
                if p1 == p2:
                    continue
                np2 = np.array(p2)
                if np.linalg.norm(np1 - np2) < 2:
                    return 0
        return 1

    
    def Q_values(self, actions: List[ActionInfo]) -> List[ActionInfo]:
        for a in actions:
            a.new_Q_value = np.dot(self.theta, a.new_features)
        return actions
    
#    def get_Q_value(self, features):
#        q_value = np.dot(self.weights, features)
#        return q_value
    
    def learn(self, prev_action: ActionInfo, actions: Optional[List[ActionInfo]], reward: float):
        logging.info(f'{self.__class__.__name__}.learn()')
        future_reward: float = 0

        if actions is not None:
            logging.info('TD')
            max_Q_value = np.max([a.new_Q_value for a in actions])
            action = random.choice([a for a in actions if a.new_Q_value == max_Q_value])

            self.history_theta.append(self.theta.copy())
            assert action.new_Q_value is not None
            future_reward = self.gamma * action.new_Q_value
        else:
            logging.info('Terminal state')

        assert prev_action.new_Q_value is not None
        assert prev_action.new_features is not None
        logging.info(f'Old theta: {self.theta}')
        self.theta += self.alfa * (reward + future_reward - prev_action.new_Q_value) * prev_action.new_features
        logging.info(f'New theta: {self.theta}')
            
    def loss(self, state):
        if self.is_learning:
            self.learn(self.history_actions[-1], None, -100)
        
    
    def win(self, state):
        if self.is_learning:
            self.learn(self.history_actions[-1], None, 100)
        
    
    def draw(self, state):
        if self.is_learning:
            self.mc_learn(self.prev_state, self.prev_action, -5)
