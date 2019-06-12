import numpy as np
from Game import Game
from ConsoleView import ConsoleView
from ConsoleHumanAgent import ConsoleHumanAgent
from QLearningApproximationAgent import QLearningApproximationAgent
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
import logging

if __name__ == "__main__":
    #g = Game(player1=ConsoleHumanAgent(), player2=ConsoleHumanAgent(), view=ConsoleView())
    logging.basicConfig(filename='debug.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

    max_row = 5
    num_feat = ((((max_row - 1) * 2) - 1) * 2) + 1
    theta = np.zeros(num_feat)
    agent1 = QLearningApproximationAgent(0.5, 0.5, EpsilonGreedyPolicy(0.05), theta=theta)
    agent2 = QLearningApproximationAgent(0.5, 0.5, EpsilonGreedyPolicy(0.05), theta=theta)    
    print(agent1.feature_names)
    g = Game(player1=agent1, player2=agent2, view=ConsoleView(), k=5, play=False, m=None, n=None)
    g.start()
    