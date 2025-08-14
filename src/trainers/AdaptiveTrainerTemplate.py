import os
from abc import abstractmethod

import torch


class AdaptiveTrainerTemplate:
    """
    Template class for adaptive trainers

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
    def selfPlay(self, game, base_model, rl_algorithm, adaptation_algorithm):
        """
        Algorithm of playing game, until will be reached a terminal state of the game

        Args:
            game (Game): Game that will be played
            base_model (nn.Module): Base mmodel of the game
            # adaptive_model (nn.Module): Adaptive model of the game, that will be trained
            rl_algorithm (): RL algorithm
            adaptation_algorithm (AdaptTemplate): algorithm for calculation new policy

        Returns:
            memory (np.array): memory of the game
        """
        pass

    @abstractmethod
    def train(self, adaptive_model, adaptive_optimizer, memory, loss_coef):
        """
        Method for changing model weights

        Args:
            # base_model (nn.Module): Base mmodel of the game
            adaptive_model (nn.Module): Adaptive model of the game, that will be trained
            adaptive_optimizer (optim.Optimizer): Optimizer that will be used for the adaptive model
            memory (np.array): memory of the games
            loss_coef (dict): Loss coefficient
        """
        pass

    @abstractmethod
    def learn(self, game, base_model, adaptive_model, adaptive_optimizer, rl_algorithm, adaptation_algorithm, loss_coef):
        """
        Method execution whole loop of training the model

        Args:
            game (Game): Game that will be played
            base_model (nn.Module): Base mmodel of the game
            adaptive_model (nn.Module): Adaptive model of the game, that will be trained
            adaptive_optimizer (optim.Optimizer): Optimizer that will be used for the adaptive model
            rl_algorithm (): RL algorithm
            adaptation_algorithm (AdaptTemplate): algorit   hm for calculation new policy
            loss_coef (dict): Loss coefficient
        """
        pass