import json
from abc import abstractmethod

import os
import torch
import torch.nn as nn

from src.games import Game


class TrainerTemplate:
    """
    This is a parent class, for algorithms

    Attributes:
        base_model (nn.Module): model that will be used for training in algorithm
        adapt_model (nn.Module): model that will adapt to opponent's level
        optimizer (torch.optim.Optimizer): optimizer that will be used for training in algorithm
        game (Game): game that will be used for training in algorithm
        algorithm (): algorithm, that will be used in training
        args ({}): arguments that will be passed to the algorithm
    """

    def __init__(self, algorithm, base_model, adapt_model, optimizer, game, args):
        """
        Constructor for initializing the AlphaZero class

        Args:
            algorithm (): algorithm that will be used for training
            base_model (nn.Module): model that will be used for training in algorithm
            adapt_model (nn.Module): model layers for weights adaptation
            optimizer (torch.optim.Optimizer): optimizer that will be used for training in algorithm
            game (Game): game that will be used for training in algorithm
            args ({}): arguments that will be passed to the algorithm
        """
        self.algorithm = algorithm
        self.base_model = base_model
        self.adapt_model = adapt_model
        self.optimizer = optimizer
        self.game = game
        self.args = args

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
        dir_path = os.path.abspath(os.getcwd())

        directory_to_save = os.path.join(dir_path, 'src/', directory_to_save)

        os.makedirs(directory_to_save, exist_ok=True)

        if self.adapt_model is None:
            result_file_name = f"{whose_weights}_{self.game.game_name.lower()}_{self.algorithm.algorithm_name.lower()}_{self.base_model.structure_name.lower()}"
        else:
            result_file_name = f"{whose_weights}_{self.game.game_name.lower()}_{self.algorithm.algorithm_name.lower()}_{self.adapt_model.structure_name.lower()}"

        if iteration is not None:
            result_file_name = result_file_name + f"_{iteration}"

        result_file_name = result_file_name + f".{file_type}"

        path = os.path.join(directory_to_save, result_file_name)

        torch.save(state_dict, path)

    def save_directories(self):
        """
        Method for identification, where models will be saved

        Returns:
            model_dir (string): directory, where model weights will be saved
            optimizers_dir (string): directory, where optimizer weights will be saved
        """
        model_dir = "base_models_weights"
        optimizer_dir = "base_optimizers_weights"

        if self.adapt_model is not None:
            model_dir = "adapt_models_weights"
            optimizer_dir = "adapt_optimizers_weights"

        return model_dir, optimizer_dir

    def load_layer(self, path):
        """
        This method load game state layers, to find equal states of the game

        Parameters:
            path (): path to the layer for loading

        Returns:
            res ({}}): dictionary for the layer
        """
        res = {}

        if not os.path.exists(path):
            return res

        with open(path, "r") as f:
            res = json.load(f)

        return res

    def save_data_in_layer(self, probs, move_values, layer=0):
        """
        This method save game state to the layer

        Parameters:
            layer (int): index of layer to check.
        """
        current_key = self.game.create_current_key()
        current_value = self.game.create_current_value(probs, move_values)

        path = os.path.dirname(os.path.abspath(__file__))
        path = path + f'/saved_states/layer_{layer}.json'

        layer = self.load_layer(path)

        layer[current_key] = current_value

        with open(path, "w") as f:
            json.dump(layer, f)

    def check_existing_layers(self, start_layer=0):
        """
        This method checking existence of states, and use them, if they are exist

        Parameters:
             start_layer (int): index of layer for the start of the search.

        Returns:
            action_probs (np.array): action probabilities
            action_values (np.array): action values
        """
        path = os.path.dirname(os.path.abspath(__file__))
        path = path + f'/saved_states/'

        current_key = self.game.create_current_key()

        for i in range(start_layer, -1, -1):
            layer_path = path + f'layer_{i}.json'

            if not os.path.exists(layer_path):
                continue

            dict = self.load_layer(layer_path)

            if current_key in dict.keys():
                return self.game.transform_value_to_data(dict[current_key])

        return None

    def clear_save_states(self):
        """
        Method for cleaning save_states directory
        """
        path = os.path.dirname(os.path.abspath(__file__))
        path = path + f'/saved_states/'

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path):  # Only delete files
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

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
