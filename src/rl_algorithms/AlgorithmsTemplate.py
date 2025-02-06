from abc import abstractmethod

import os
import torch
import torch.nn as nn

from src.games import Game


class AlgorithmsTemplate:
    """
    This is a parent class, for algorithms

    Attributes:
        model (nn.Module): model that will be used for training in algorithm
        optimizer (torch.optim.Optimizer): optimizer that will be used for training in algorithm
        game (Game): game that will be used for training in algorithm
        args ({}): arguments that will be passed to the algorithm
        algorithm_name (str): name of the algorithm
    """

    model = None
    optimizer = None
    game = None
    args = None
    algorithm_name = None

    def save_weights(self, state_dict, directory_to_save, whose_weights, file_type, iteration=None):
        """
        Method for saving the weights of the model or optimizer

        Parameters:
        state_dict (dict): state dict to be saved
        directory_to_save (string): directory to save the weights to
        whose_weights (string): whose weights will be saved
        file_type (string): file type of the weights
        iteration (string): iteration name, of the step
        """
        os.makedirs(directory_to_save, exist_ok=True)

        result_file_name = f"{whose_weights}_{self.game.game_name.lower()}_{self.algorithm_name.lower()}_{self.model.structure_name.lower()}"

        if iteration is not None:
            result_file_name = result_file_name + f"_{iteration}"

        result_file_name = result_file_name + f".{file_type}"

        path = os.path.join(directory_to_save, result_file_name)

        torch.save(state_dict, path)

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
        Files should contain the name of the game, the name of the algorithm, and the structure used, all in lowercase format.
        """
        pass
