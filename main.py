from Game import Game
from ConsoleView import ConsoleView
from ConsoleHumanAgent import ConsoleHumanAgent
from QLearningApproximationAgent import QLearningApproximationAgent
from EpsilonGreedyChooser import EpsilonGreedyChooser
import numpy as np

if __name__ == "__main__":
    #g = Game(player1=ConsoleHumanAgent(), player2=ConsoleHumanAgent(), view=ConsoleView())
    max_row = 5
    num_feat = ((((max_row - 2) * 2) + 1) * 2) - 1
    weights = np.zeros(num_feat)
    agent1 = QLearningApproximationAgent(0.5, 0.5, EpsilonGreedyChooser(0.05), weights=weights)
    agent2 = QLearningApproximationAgent(0.5, 0.5, EpsilonGreedyChooser(0.05), weights=weights)    
    g = Game(player1=agent1, player2=agent2, view=ConsoleView())