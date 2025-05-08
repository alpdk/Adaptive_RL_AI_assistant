import math
from abc import abstractmethod

import torch
import numpy as np

from src.rl_algorithms.base_rl_algorithms.Node import Node

class BaseProbsTemplate:
    """
    Template class for realization of different RL algorithms for calculating move probabilities.

    Attributes:
        game (Game): Game that will be played.
        args ({}): some arguments that will be passed to the MCTS algorithm.
        model (nn.Module): the model that will be trained.
        algorithm_name (str): name of the algorithm.
    """
    def __init__(self, game, args, model):
        """
        Constructor.

        Parameters:
            game (Game): Game that will be played.
            args ({}): some arguments that will be passed to the RL algorithm.
            model (nn.Module): the model that will be trained.
        """
        self.game = game
        self.args = args
        self.model = model
        self.algorithm_name = None

    def calc_probs(self, root, values=None):
        """
        Method for calculating the probabilities of each action

        Parameters:
             root (Node): root of the current game
             values (np.array[float]): array of values

        Returns:
            np.array[]: probabilities of each action
        """
        action_probs = np.zeros(self.game.action_size)
        min_val = np.min(values)
        max_val = np.max(values)

        for child in root.children:
            action = child.action_taken
            # action_probs[action] = math.exp(values[action])
            action_probs[action] = math.pow(values[action] + 1, 2) / (1 - values[action] + 1e-5)
        action_probs /= np.sum(action_probs)

        return action_probs

    def calc_values(self, root):
        """
        Method for calculating the values of each action

        Parameters:
             root(Node): root of the current game

        Returns:
            np.array[]: values of each action
        """
        action_values = np.zeros(self.game.action_size)
        for child in root.children:
            action = child.action_taken
            action_values[action] = child.value_sum / child.visit_count

        return action_values

    @abstractmethod
    @torch.no_grad()
    def search(self, player):
        """
        Method that performs MCTS search.

        Args:
            player (int): current player

        Returns:
            action (np.array): action probabilities
        """
        pass