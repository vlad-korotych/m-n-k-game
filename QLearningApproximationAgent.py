from base import Agent, Policy, Mark, Action, GameState, FeaturesModel
import numpy as np
from typing import List, Optional, IO
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
                 reg: int = 1000,
                 learning: bool = True,
                 theta: Optional[np.ndarray] = None,
                 inf_field: bool = True,
                 debug: bool = False
                 ):
        self.alfa: float = alfa
        self.gamma: float = gamma
        self.policy: Policy = policy
        self.features_model = features_model
        self.is_learning: bool = learning
        self.row: int = row
        self.history_actions: List[ActionInfo] = []
        self.history_theta: List[np.ndarray] = []
        self.inf_field: bool = inf_field
        self.debug = debug
        self.reg = reg

        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.zeros(self.features_model.features_count() + 1)

    def get_action(self, new_state: GameState) -> Action:
        if self.debug:
            logging.info(f'{self.__class__.__name__}.get_action()')

        actions = self.get_possible_actions(new_state)
        if self.debug:
            logging.info(f'Player: {new_state.current_player.name}')
            logging.info(f'\nPossible actions: {len(actions)}')

        Q_table = new_state.board.copy().astype(dtype=np.object)
        Q_table[Q_table == Mark.X.value] = Mark.X.name
        Q_table[Q_table == Mark.O.value] = Mark.O.name
        for a in actions:
            if self.debug:
                logging.info(f'Action: {a.action}')
            a.new_board = new_state.board.copy()
            a.new_board[a.action.row, a.action.col] = new_state.current_player.value
            if self.debug:
                logging.info(f'Board:\n{np.array2string(a.new_board, max_line_width=np.inf)}')
            a.new_features = np.insert(self.features_model.get_features(GameState(a.new_board, new_state.current_player, new_state.is_game_end, new_state.winner)), 0, 1)

            if self.debug:
                logging.info(f'Features: {len(a.new_features)} {np.array2string(a.new_features, max_line_width=np.inf)}')
            a.new_Q_value = np.dot(self.theta, a.new_features)
            if self.debug:
                logging.info(f'Q value: {a.new_Q_value}')
            Q_table[a.action.row, a.action.col] = a.new_Q_value

        if self.debug:
            logging.info(f'Q table:\n{np.array2string(Q_table, max_line_width=np.inf)}')

        if len(self.history_actions) > 0 and self.is_learning is not None:
            reward = -1
            self.learn(self.history_actions[-1], actions, reward)

        action = self.policy.get_action(new_state, actions)

        if self.debug:
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
        if self.debug:
            logging.info(f'{self.__class__.__name__}.learn()')
        future_reward: float = 0

        if actions is not None:
            if self.debug:
                logging.info('TD')
            max_Q_value = np.max([a.new_Q_value for a in actions])
            action = random.choice([a for a in actions if a.new_Q_value == max_Q_value])

            assert action.new_Q_value is not None
            future_reward = self.gamma * action.new_Q_value
        elif self.debug:
            logging.info('Terminal state')

        self.history_theta.append(self.theta.copy())
        assert prev_action.new_Q_value is not None
        assert prev_action.new_features is not None
        if self.debug:
            logging.info(f'Old theta: {self.theta}')
        expected = reward + future_reward
        current = prev_action.new_Q_value
#        grad = -(expected - current) * prev_action.new_features
        X = prev_action.new_features
        grad = -(expected - current) * np.dot(np.linalg.inv(np.dot(X.T, X) + self.reg * np.identity(len(X.T))), X.T).T
        if self.debug:
            logging.info(f'expected: {expected}, current: {current}, features: {prev_action.new_features}')
        self.theta -= self.alfa * grad
        if self.debug:
            logging.info(f'New theta: {self.theta}')

    def loss(self, state):
        if self.is_learning:
            self.learn(self.history_actions[-1], None, -100)

    def win(self, state):
        if self.is_learning:
            self.learn(self.history_actions[-1], None, 100)

    def draw(self, state):
        if self.is_learning:
            self.learn(self.history_actions[-1], None, -20)

    def get_learn_params(self) -> str:
        return str(list(self.theta))
