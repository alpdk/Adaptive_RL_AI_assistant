import argparse
import os

import torch

from src.games.UltimateTicTacToe import UltimateTicTacToe
from src.models_structures.ModifiedResNet import ModifiedResNet
from src.rl_algorithms.MCTS import MCTS
from src.trainers.Trainer import Trainer

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


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# game = UltimateTicTacToe()
# model = ModifiedResNet(game, 9, 128, device=device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
#
# model_weights = "modified_model_UTTT.pt"
# # optimizer_weights = "modified_optimizer_UTTT.pt"
#
# path = os.path.dirname(os.path.abspath(__file__))
# model_path = path + f'/weights/{model_weights}'
#
# model.load_state_dict(torch.load(path, weights_only=True))
#
# freeze_list = ["startBlock", "backBone", "policyHead", "valueHead"]
#
# for name, param in model.named_parameters():
#     if any(target in name for target in freeze_list):
#         param.requires_grad = False
#         print(f"Froze: {name}")
#
# algorithms = MCTS(train_args)
# trainer = Trainer(train_args)
#
# loss_coef = {'policy_loss': 0.0,
#              'moves_values_loss': 1.0,
#              'value_loss': 0.0}
#
# trainer.learn(game, model, optimizer, algorithms, loss_coef)

def parse_base_model_arguments():
    """
    Parse command line arguments

    Returns:
         args (Namespace): parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for training a model')

    parser.add_argument('model_weights', type=str,
                        help='Name of the file with model weights')

    parser.add_argument('optimizer_weights', type=str,
                        help='Name of the file with optimizer weights')

    return parser.parse_args()

def freeze_layers(model, layers):
    """
    Method for freezing layers in model

    Args:
        model (nn.Module): model
        layers (list): layers that should be freezed
    """

    for name, param in model.named_parameters():
        if any(target in name for target in layers):
            param.requires_grad = False
            print(f"Froze: {name}")


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
    """
    Main function
    """
    args = parse_base_model_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game = UltimateTicTacToe()
    model = ModifiedResNet(game, 9, 128, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    load_weights(model, args.model_weights, optimizer, args.optimizer_weights)

    freeze_list = ["startBlock", "backBone", "policyHead", "valueHead"]
    freeze_layers(model, freeze_list)

    algorithms = MCTS(train_args)
    trainer = Trainer(train_args)

    loss_coef = {'policy_loss': 0.0,
                 'moves_values_loss': 1.0,
                 'value_loss': 0.0}

    trainer.learn(game, model, optimizer, algorithms, loss_coef)


if __name__ == '__main__':
    main()
