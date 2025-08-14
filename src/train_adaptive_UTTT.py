import argparse
import torch
import os

from src.adaptation_algorithms.ReverseSquare import ReverseSquare
from src.games.UltimateTicTacToe import UltimateTicTacToe
from src.models_structures.BaseLinear import BaseLinear
from src.models_structures.ModifiedResNet import ModifiedResNet
from src.rl_algorithms.AdaptiveRL import AdaptiveRL
from src.trainers.AdaptTrainer import AdaptTrainer

train_args = {
    'C': 0.57,
    'tau': 0.737,
    'num_iterations': 1,
    'num_searches': 100,
    'num_selfPlay_iterations': 1,
    'num_epochs': 100,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'history_depth': 5,
    'relevance_coef': 1.0
}


def parse_arguments():
    """
    Parse command line arguments

    Returns:
         args (Namespace): parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for training a model')

    parser.add_argument('base_model_weights', type=str,
                        help='Name of the file with model weights for the base model')

    # parser.add_argument('base_optimizer_weights', type=str,
    #                     help='Name of the file with optimizer weights for the base model')

    parser.add_argument('adaptive_model_weights', type=str,
                        help='Name of the file with model weights for the adaptive model')

    parser.add_argument('adaptive_optimizer_weights', type=str,
                        help='Name of the file with optimizer weights for the adaptive model')

    return parser.parse_args()


def load_weights(model, model_weights, optimizer, optimizer_weights):
    """
    Method for loading weights for a model and an optimizer

    Args:
        model (nn.Module): model
        model_weights (str): name of the file with model weights
        optimizer (): optimizer
        optimizer_weights (str): name of the file with optimizer weights
    """
    path = os.path.dirname(os.path.abspath(__file__))

    model_path = path + f'/weights/model/{model_weights}'
    optimizer_path = path + f'/weights/optimizer/{optimizer_weights}'

    if model_weights != "None":
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            print(f"There is no such model weights file: {model_weights}")

    if optimizer_weights != "None":
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
        else:
            print(f"There is no such optimizer weights file: {optimizer_weights}")


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game = UltimateTicTacToe()

    base_model = ModifiedResNet(game, 9, 128, device=device)
    # base_optimizer = torch.optim.AdamW(base_model.parameters(), lr=0.001, weight_decay=0.0001)

    adaptive_model = BaseLinear(game, 6, 128, device=device)
    adaptive_optimizer = torch.optim.AdamW(base_model.parameters(), lr=0.001, weight_decay=0.0001)

    load_weights(base_model, args.base_model_weights, None, "None")
    load_weights(adaptive_model, args.adaptive_model_weights, adaptive_optimizer, args.adaptive_optimizer_weights)

    adaptation_algorithm = ReverseSquare(train_args)
    rl_algorithms = AdaptiveRL(train_args)
    trainer = AdaptTrainer(train_args)

    loss_coef = {'policy_loss': 1.0,}

    trainer.learn(game, base_model, adaptive_model, adaptive_optimizer, rl_algorithms, adaptation_algorithm, loss_coef)


if __name__ == '__main__':
    main()
