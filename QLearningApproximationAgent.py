from base import Agent, Chooser, Mark
import numpy as np
from typing import List, Optional
#import copy
import random

class QLearningApproximationAgent(Agent):
    def __init__(self, alfa: float, gamma: float, chooser: Chooser, row: int = 5, learning: bool = True, weights: Optional[List[float]]=None, inf_field: bool = True):
        self.alfa = alfa
        self.gamma = gamma
        self.chooser = chooser
        self.is_learning = learning
        self.row = row
        self.prev_state = None
        self.prev_action = None
        self.inf_field = inf_field
        
        
        self.features_count = ((((row - 2) * 2) + 1) * 2) - 1
        # for example, current player is X, and features will be count of each:
        # XXXXX, XXXX , OXXXX , XXX  , OXXX  , XX   , OXX   ,
        # OOOO , XOOOO , OOO  , XOOO  , OO   , XOO   .

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.zeros(self.features_count)
    
    def get_action(self, state):
        possible_actions = self.get_possible_actions(state)
        features = self.get_actions_features(state, possible_actions)
        
        if self.prev_state is not None and self.is_learning:
            reward = self.get_reward(self.prev_state, self.prev_action)
            self.td_learn(self.prev_state, self.prev_action, features, reward)
        
        weighted_actions = self.weigh_actions(features)
            
        action = self.chooser.choose_action(state, weighted_actions)
        
        self.prev_state = state
        self.prev_action = action[0]
        
        return action[0]
    
    def get_actions_features(self, state, possible_actions):
        features = []
        for action in possible_actions:
            #new_state = copy.deepcopy(state)
            #new_state.board[action[0], action[1]] = new_state.current_player.value
            state.board[action[0], action[1]] = state.current_player.value
            features.append((action, self.get_features(state)))
            # revert action
            state.board[action[0], action[1]] = Mark.NO.value
        return features
        
    
    
    def get_reward(self, prev_state, prev_action):
        if self.inf_field:
            # if board is infinite, real board will grow to infinity too
            # so, we need, that the agent move marks to center
            h, w = prev_state.board.shape
            v = np.where(prev_state.board != 0)
            center = np.array([np.mean(v[0]), np.mean(v[1])])
            d = np.linalg.norm(np.array([prev_action[0], prev_action[1]]) - center)
            if d < 10:
                return -1
            else:
                return np.floor(10 - d)
        else:
            return -1
    
    def get_possible_actions(self, state):
        f = np.where(state.board == Mark.NO.value)
        possible_actions = list(zip(f[0], f[1]))
        return possible_actions
    
    def get_features(self, state):

        no_value = Mark.NO.value

        directions = [(0, 1), (1, 1), (1, 0), (1, -1)]
        b = state.board
        h, w = b.shape
        #print(f'h, w: {h}, {w}')
        features = {}
        for i in range(2, self.row):
            features[(1, i, True)] = 0
            features[(2, i, True)] = 0
            features[(1, i, False)] = 0
            features[(2, i, False)] = 0
        features[(1, self.row, False)] = 0
        #rows = []
        v = np.zeros((h, w, len(directions)))
        
        for r in range(h):
            for c in range(w):
                #print(f'r, c, val: {r}, {c}, {b[r, c]}')
                if b[r, c] == no_value: # empty cell, next
                    continue
                for d_i, (d_r, d_c) in enumerate(directions):
                    #print(f'direction: {(d_r, d_c, d_i)}')
                    if v[r, c, d_i] == 1: # we already count this cell in this direction
                        #print('already count')
                        continue
                    m = b[r, c] # memorize mark
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
                        
                        nm = b[nr, nc]
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
                            
                        nm = b[nr, nc]
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
                        if m == state.current_player.value:
                            #rows.append((row, (1, cnt, enemy_behind)))
                            features[(1, cnt, enemy_behind)] += 1
                        else:
                            #rows.append((row, (2, cnt, enemy_behind)))
                            features[(2, cnt, enemy_behind)] += 1
        #print(rows)
        return list(features.values())
    
    def weigh_actions(self, features):
        weighted_actions = []
        for action, feat in features:
            weighted_actions.append((action, self.get_Q_value(feat)))
            
        return weighted_actions
    
    def get_Q_value(self, features):
        q_value = np.dot(self.weights, features)
        return q_value
    
    def td_learn(self, prev_state, prev_action, features, reward):
        weighted_actions = self.weigh_actions(features)
        
        max_weight = np.max([weight for action, weight in weighted_actions])
        action = random.choice([(action, weight) for action, weight in weighted_actions if weight == max_weight])
        q_value = action[1]
        expected = reward + self.gamma * q_value
        
        prev_state.board[prev_action[0], prev_action[1]] = prev_state.current_player.value
        prev_feat = self.get_features(prev_state)
        current = self.get_Q_value(prev_feat)
        
        td = expected - current
        
        for i in range(self.features_count):
            self.weights[i] += self.alfa * td * prev_feat[i]
            
    def mc_learn(self, prev_state, prev_action, reward):
        prev_state.board[prev_action[0], prev_action[1]] = prev_state.current_player.value
        prev_feat = self.get_features(prev_state)
    
        q_value = self.get_Q_value(prev_feat)
        for i in range(self.features_count):
            self.weights[i] += self.alfa * (reward - q_value) * prev_feat[i]
        
    
    def loss(self, state):
        if self.is_learning:
            self.mc_learn(self.prev_state, self.prev_action, -100)
        
    
    def win(self, state):
        if self.is_learning:
            self.mc_learn(self.prev_state, self.prev_action, 100)
        
    
    def draw(self, state):
        if self.is_learning:
            self.mc_learn(self.prev_state, self.prev_action, -5)
