import numpy as np
from Game import Game
from ConsoleView import ConsoleView
from ConsoleHumanAgent import ConsoleHumanAgent
from QLearningApproximationAgent import QLearningApproximationAgent
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
from NumRowsFeatures import NumRowsFeatures
from CartNumRowsFeatures import CartNumRowsFeatures
import logging

if __name__ == "__main__":
    #g = Game(player1=ConsoleHumanAgent(), player2=ConsoleHumanAgent(), view=ConsoleView())
    logging.basicConfig(filename='debug.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

    #feat_model = NumRowsFeatures(row=5)
    feat_model = CartNumRowsFeatures(row=5)
    num_feat = feat_model.features_count()
    theta = np.zeros(num_feat)
    agent1 = QLearningApproximationAgent(0.5, 0.5, EpsilonGreedyPolicy(0.05), theta=theta, features_model=feat_model)
    agent2 = QLearningApproximationAgent(0.5, 0.5, EpsilonGreedyPolicy(0.05), theta=theta, features_model=feat_model)
    print(feat_model.feature_names)
    g = Game(player1=agent1, player2=agent2, view=ConsoleView(), k=5, play=False, m=None, n=None)
    g.start()
    