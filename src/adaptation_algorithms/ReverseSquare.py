import math

import numpy as np

from src.adaptation_algorithms.AdaptTemplate import AdaptTemplate


class ReverseSquare(AdaptTemplate):
    """
    This class implement the ide of using the reverse square to identify  more relevant moves

    Attributes:
        args (dict): dictionary of arguments
    """

    def __init__(self, args):
        """
        Constructor

        Args:
            args (dict): dictionary of arguments
        """
        super().__init__(args)

    def _calc_balance(self, game, history):
        """
        Method for calculating the balance of the game

        Args:
            game (Game): game that we play
            history (list): history of the game

        Returns:
            balance (float): balance of the game
        """
        if len(history) == 0:
            return 1.0

        current_player = game.logger.current_player

        opponent_history = []

        for action in history:
            if action.parent.current_player != current_player:
                opponent_history.append(action)

        current_balance = 0

        for action in opponent_history:
            move_values = action.parent.action_values
            selected_move = action.changes['move_index']

            current_balance += move_values[selected_move]

        return current_balance / len(opponent_history)



    def update_probs(self, game, policy, values):
        """
        Method for updating the probabilities of each action

        Args:
            game (Game): game that we play
            policy (np.array): array of probabilities for each action
            values (np.array): array of values for each action

        Returns:
            new_policy (np.array): array of new probabilities for each action
            target_value (np.float): target value for highest probability
        """
        history = self._history_collection(game)
        target_value = self._calc_balance(game, history)

        new_policy = np.zeros_like(policy)

        for i in range(len(policy)):
            if policy[i] != 0:
                new_policy[i] = 1 / max(0.01, math.pow(target_value - values[i], 2))

        new_probs = new_policy / np.sum(new_policy)

        return new_probs, target_value

