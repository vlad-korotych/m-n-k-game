from base import Agent, Policy, Mark, Action, GameState, FeaturesModel
import numpy as np
from typing import List, Optional, Dict, NamedTuple, Union, Tuple
import logging
import random
from dataclasses import dataclass

@dataclass
class ActionInfo:
    action: Action
    new_board: Optional[np.ndarray] = None
    new_features: Optional[np.ndarray] = None
    new_Q_value: Optional[float] = None

class QLearningApproximationAgent(Agent):
    def __init__(self,
                 alfa: float,
                 gamma: float,
                 policy: Policy,
                 features_model: FeaturesModel,
                 row: int = 5,
                 learning: bool = True,
                 theta: Optional[np.ndarray]=None,
                 inf_field: bool = True):
        self.alfa: float = alfa
        self.gamma: float = gamma
        self.policy: Policy = policy
        self.features_model = features_model
        self.is_learning: bool = learning
        self.row: int = row
        self.history_actions: List[ActionInfo] = []
        self.history_theta: List[np.ndarray] = []
        self.inf_field: bool = inf_field
        
        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.zeros(self.features_model.features_count())

    
    def get_action(self, new_state: GameState) -> Action:
        logging.info(f'{self.__class__.__name__}.get_action()')
        actions = self.get_possible_actions(new_state)
        logging.info(f'Player: {new_state.current_player.name}')
        logging.info(f'\nPossible actions: {len(actions)}')

        actions = self.get_actions_features(new_state, actions)

        actions = self.Q_values(actions)

        Q_table = new_state.board.copy().astype(dtype=np.object)
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
            a.new_features = self.features_model.get_features(GameState(a.new_board, state.current_player, state.is_game_end, state.winner))
        return actions
    
    def get_possible_actions(self, state: GameState) -> List[ActionInfo]:
        f = np.where(state.board == Mark.NO.value)
        possible_actions = [ActionInfo(action=Action(r, c)) for r, c in zip(f[0], f[1])]
        return possible_actions

    def Q_values(self, actions: List[ActionInfo]) -> List[ActionInfo]:
        for a in actions:
            a.new_Q_value = np.dot(self.theta, a.new_features)
        return actions
    
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
