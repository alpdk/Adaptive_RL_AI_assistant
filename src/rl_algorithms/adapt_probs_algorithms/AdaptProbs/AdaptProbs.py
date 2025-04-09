import sys
import math

import numpy as np

from src.rl_algorithms.adapt_probs_algorithms.AdaptProbsTemplate import AdaptProbsTemplate

class AdaptProbs(AdaptProbsTemplate):
    """
    This class implements adaptation algorithm for move probabilities.

    Attributes:
        game (Game): game, that is played with history
        history_depth (int): how many steps, will be chosen for current one
        algorithm_name (str): name of the algorithm.
        extra_data ({}): dictionary of additional args:
                                {
                                    'rel_coef' (float): how relevant should be moved for
                                                        calculation of median value
                                }
    """

    def __init__(self, game, history_depth, extra_data):
        """
        Constructor.

        Parameters:
            game (Game): game, that is played with history
            history_depth (int): how many steps, will be chosen for current one
            extra_data ({}): dictionary of additional args
        """

        super().__init__(game, history_depth, extra_data)
        self.algorithm_name = "AdaptProbs"

    def calc_player_income(self, opponent_moves):
        """
        Method for calculating opponent player's income for last n moves.

        Parameters:
             opponent_moves (np.ndarray): opponent moves

        Returns:
            int: median income by player
        """
        median_opponent_income = 0
        divider = 0

        if opponent_moves[0] == None:
            return 1

        for i in range(len(opponent_moves)):
            if opponent_moves[i] is None:
                break

            last_action = opponent_moves[i].child.last_action

            min_val, max_val = sys.float_info.max, -sys.float_info.max

            for j in range(len(opponent_moves[i].action_values)):
                if opponent_moves[i].action_probs[j] != 0:
                    min_val = min(min_val, opponent_moves[i].action_values[j])
                    max_val = max(max_val, opponent_moves[i].action_values[j])

            if min_val == max_val:
                return 1.0

            median_opponent_income += (math.pow(self.extra_data['rel_coef'], i) *
                                       (opponent_moves[i].action_values[last_action] -
                                        min_val) /
                                       (max_val -
                                        min_val))

            divider += math.pow(self.extra_data['rel_coef'], i)

        if divider != 0:
            median_opponent_income /= divider

        return median_opponent_income

    # def calc_player_income(self, opponent_moves):
    #     """
    #     Method for calculating opponent player's income for last n moves.
    #
    #     Parameters:
    #          opponent_moves (np.ndarray): opponent moves
    #
    #     Returns:
    #         int: median income by player
    #     """
    #     median_opponent_income = 0
    #     divider = 0
    #
    #     if opponent_moves[0] == None:
    #         return 1
    #
    #     for i in range(len(opponent_moves)):
    #         if opponent_moves[i] is None:
    #             break
    #
    #         last_action = opponent_moves[i].child.last_action
    #
    #         median_opponent_income += opponent_moves[i].action_values[last_action] * math.pow(self.extra_data['rel_coef'], i)
    #         divider += 1
    #
    #     return median_opponent_income / divider

    def probs_modification(self):
        """
        Method for probs modification

        Returns:
            median_opponent_income (int): median income by player
            new_probs (np_array()): new probabilities
        """
        opponent_moves = self.game.collect_opponent_moves(self.history_depth)

        if opponent_moves[0] is None:
            return 1, self.game.logger.action_probs

        median_opponent_income = self.calc_player_income(opponent_moves)

        new_probs = self.game.logger.action_probs.copy()

        min_val, max_val = sys.float_info.max, sys.float_info.min

        for i in range(len(self.game.logger.action_values)):
            if self.game.logger.action_probs[i] != 0:
                min_val = min(min_val, self.game.logger.action_values[i])
                max_val = max(max_val, self.game.logger.action_values[i])

        if min_val == max_val:
            return 1, self.game.logger.action_probs

        for i in range(len(self.game.logger.action_values)):
            if self.game.logger.action_values[i] != 0:
                median_income = ((self.game.logger.action_values[i] - min_val) /
                                 (max_val - min_val))
                new_probs[i] = 1.0 / (abs(median_income - median_opponent_income) + 1e-3)

        # for i in range(len(self.game.logger.action_values)):
        #     if self.game.logger.action_values[i] != 0:
        #         new_probs[i] = 1.0 / (abs(self.game.logger.action_values[i] - median_opponent_income) + 1e-3)

        new_probs = new_probs / np.sum(new_probs)

        return median_opponent_income, new_probs
