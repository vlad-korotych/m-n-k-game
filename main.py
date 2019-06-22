import numpy as np
from Game import Game
from ConsoleView import ConsoleView
from ConsoleHumanAgent import ConsoleHumanAgent
from QLearningApproximationAgent import QLearningApproximationAgent
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
from NumRowsFeatures import NumRowsFeatures
from CartNumRowsFeatures import CartNumRowsFeatures
#import logging
from tqdm import tqdm
import argparse
from typing import Optional, IO

if __name__ == "__main__":
    #g = Game(player1=ConsoleHumanAgent(), player2=ConsoleHumanAgent(), view=ConsoleView())
    #logging.basicConfig(filename='debug.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description='m,n,k-game')
    parser.add_argument("-l", "--log_file", help="Specify log file", type=str)
    parser.add_argument("-g", "--games_count", help="Number of games", type=int)
    args = parser.parse_args()
    log_filename = args.log_file
    log_file: Optional[IO] = None
    if log_filename is not None:
        log_file = open(log_filename, 'w')
        title = "game_no;turn;player;height;width;learn_params;row;column\n"
        log_file.write(title)

    games_cnt = 1
    if args.games_count is not None:
        games_cnt = args.games_count

    try:
        max_row = 5
        feat_model = NumRowsFeatures(row=5)
        #feat_model = CartNumRowsFeatures(row=5)
        num_feat = feat_model.features_count() + 1
        theta = np.zeros(num_feat)
        for i in tqdm(range(games_cnt)):
            print(f'Game {i}')
            print(f'Theta1 {theta}')
            
            agent1 = QLearningApproximationAgent(0.5, 0.5, EpsilonGreedyPolicy(0.05), theta=theta, features_model=feat_model, inf_field=True)
            agent2 = QLearningApproximationAgent(0.5, 0.5, EpsilonGreedyPolicy(0.05), theta=theta, features_model=feat_model, inf_field=True)
            g = Game(player1=agent1, player2=agent2, view=None, m=None, n=None, log_file=log_file, game_no=i)
    finally:
        if log_file is not None:
            log_file.close()
