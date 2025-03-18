import os
import torch
import argparse

from src.games.Checkers import Checkers
from src.games.Game import Game
from src.models_structures.adaptive_play.BaseLinear import BaseLinear
from src.models_structures.base_play.ResNet import ResNet
from src.rl_algorithms.adapt_probs_algorithms.AdaptProbs.AdaptProbs import AdaptProbs
from src.rl_algorithms.AdaptTrainer import AdaptTrainer
from src.rl_algorithms.BaseTrainer import BaseTrainer
from src.rl_algorithms.base_rl_algorithms.MCTS.MCTS import MCTS


def load_game(name: str):
    """
    Load game, that will be played between models

    Parameters:
         name (str): Name of the game

    Returns:
        game (Game): Game instance
    """
    print(f"Loading {name} game...")

    name = name.lower()

    match name:
        case "checkers":
            return Checkers(4, 4)
        case _:
            return None


def load_model_structure(model_structure: str, game: Game, device: torch.device):
    """
    Load model structure, that will be played between models

    Parameters:
         model_structure (str): Name of model structure
         game (Game): Game instance
         device (torch.device): Device to use

    Returns:
        model (nn.Module): Model structure instance
    """

    model_structure = model_structure.lower()

    match model_structure:
        case "resnet":
            return ResNet(game, 4, 64, device)
        case "baselinear":
            return BaseLinear(game, 4, 64, device)
        case _:
            return None

def load_rl_algorithm(algorithm_name, game, args, model):

    algorithm_name = algorithm_name.lower()

    match algorithm_name:
        case "mcts":
            return MCTS(game, args, model)
        case _:
            return None

def load_trainer(trainer_name, algorithm, base_model, adapt_model, optimizer, game, train_args):
    """
    Method for loading a RL algorithm, that will be used in training model

    Args:
        algorithm (TrainerTemplate): Algorithm that will be used in training
        base_model (torch.nn.Module): Base model for taking move decisions
        adapt_model (torch.nn.Module): Model for adapting to opponent level
        optimizer (torch.optim.Optimizer): Optimizer for training
        rl_algorithm_name (str): Name of RL algorithm
        train_args ({}): Dictionary of arguments passed to training

    Returns:
         res (): Class for training models
    """
    trainer_name = trainer_name.lower()

    match trainer_name:
        case 'basetrainer':
            return BaseTrainer(algorithm, base_model, None, optimizer, game, train_args)
        case 'adapttrainer':
            return AdaptTrainer(algorithm, base_model, adapt_model, optimizer, game, train_args)
        case _:
            return None


def load_model(device: torch.device,
               game: Game,
               directory: str,
               model_name: str,
               model_structure: str):
    """
    Load model from directory

    Parameters:
        directory (str): Directory of the model
        model_name (str): Name of the model
        model_structure (str): Name of the model structure
        game (Game): Game instance
        device (torch.device): Device to use

    Returns:
        model (Model): Model with loaded weights
    """
    model_weights = "model_" + game.game_name.lower() + "_" + model_name.lower() + "_" + model_structure.lower() + ".pth"

    path = os.path.dirname(os.path.abspath(__file__))
    path = path + f'/{directory}/{model_weights}'

    if not os.path.exists(path):
        print(
            f'Model trained for game \"{game.game_name.lower()}\", algorithm \"{model_name.lower()}\", and model structure \"{model_structure.lower()}\" does not exist inside of the \"{directory}/\" directory!')
        return None

    model = load_model_structure(model_structure, game, device)

    if model is None:
        print(f"There are no such model structure: {model_structure}!!!")
        return None

    model.load_state_dict(torch.load(path, weights_only=True))

    return model


def load_adapt_algortihm(adaptive_algorithm_name, game, history_depth, extra_data):
    """
    Method for loading adaptive algorithm

    Args:
        adaptive_algorithm_name (str): Name of adaptive algorithm
        game (str): Game that will be played
        history_depth (int): how many opponent last moves will be used for calculation of opponent median income coefficient
        extra_data ({}}): Dictionary of extra data for adapt probability method
    """
    adaptive_algorithm_name = adaptive_algorithm_name.lower()

    " !!! For some time, but have to be removed !!! "
    extra_data = {"rel_coef": 1.0}

    match adaptive_algorithm_name:
        case "adaptprobs":
            return AdaptProbs(game, history_depth, extra_data)
        case _:
            return None
