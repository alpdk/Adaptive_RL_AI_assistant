from abc import abstractmethod

import torch
import torch.nn as nn

from src.games import Game

class ModelTemplates:
    """
    This is a parent class, for models

    Attributes:
        model (nn.Module): model that will be used for training in algorithm
        optimizer (torch.optim.Optimizer): optimizer that will be used for training in algorithm
        game (Game): game that will be used for training in algorithm
        args ({}): arguments that will be passed to the algorithm
        model_name (str): name of the model
    """

    model = None
    optimizer = None
    game = None
    args = None
    model_name = None

    @abstractmethod
    def selfPlay(self):
        """
        Algorithm of playing game, until will be reached a terminal state of the game

        Returns:
            memory (np.array): memory of the game
        """
        pass

    @abstractmethod
    def train(self, memory):
        """
        Training our model on a base of played games played

        Args:
            memory (np.array): memory of the games
        """
        pass

    @abstractmethod
    def learn(self):
        """
        Whole process of learning model on a base of played games

        In the end model and optimizer should be saved in directories "trained_models" and "trained_optimizers"
        Files should contain name of the model, and used structure.
        """
        pass