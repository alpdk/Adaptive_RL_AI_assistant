import os
from abc import abstractmethod

import torch


class BaseTrainerTemplate:
    """
    Template class for base trainers

    Attributes:
        args ({}): Dictionary of arguments, that will be used for training
    """

    def __init__(self, args):
        """
        Constructor

        Args:
            args ({}): Dictionary of arguments, that will be used for training
        """
        self.args = args

    def save_directories(self):
        """
        Method for identification, where models will be saved

        Returns:
            model_dir (string): directory, where model weights will be saved
            optimizers_dir (string): directory, where optimizer weights will be saved
        """
        model_dir = "weights/model"
        optimizer_dir = "weights/optimizer"

        return model_dir, optimizer_dir

    def save_weights(self, game_name, algorithm_name, structure_name, state_dict, directory_to_save, file_type, iteration=None):
        """
        Method for saving the weights of the model or optimizer

        Parameters:
            game_name (string): Name of the game
            algorithm_name (string): Name of the algorithm
            structure_name (string): Name of the model
            state_dict (dict): state dict to be saved
            directory_to_save (string): directory to save the weights to
            file_type (string): file type of the weights
            iteration (int): iteration name, of the step
        """
        dir_path = os.path.abspath(os.getcwd())

        directory_to_save = os.path.join(dir_path, 'src/', directory_to_save)

        os.makedirs(directory_to_save, exist_ok=True)

        result_file_name = f"base_{game_name.lower()}_{algorithm_name.lower()}_{structure_name.lower()}"

        if iteration is not None:
            result_file_name = result_file_name + f"_{iteration}"

        result_file_name = result_file_name + f".{file_type}"

        path = os.path.join(directory_to_save, result_file_name)

        torch.save(state_dict, path)

    @abstractmethod
    def selfPlay(self, game, model, algorithm):
        """
        Algorithm of playing game, until will be reached a terminal state of the game

        Args:
            game (Game): Game that will be played
            model (nn.Module): Model that will be trained
            algorithm (): RL algorithm

        Returns:
            memory (np.array): memory of the game
        """
        pass

    @abstractmethod
    def train(self, model, optimizer, memory, loss_coef):
        """
        Method for changing model weights

        Args:
            model (nn.Module): Model that will be trained
            optimizer (optim.Optimizer): Optimizer that will be used
            memory (np.array): memory of the games
            loss_coef (dict): Loss coefficient
        """
        pass

    @abstractmethod
    def learn(self, game, model, optimizer, algorithm, loss_coef):
        """
        Method execution whole loop of training the model

        Args:
            game (Game): Game that will be played
            model (nn.Module): Model that will be trained
            optimizer (optim.Optimizer): Optimizer that will be used
            algorithm (): RL algorithm
            loss_coef (dict): Loss coefficient
        """
        pass