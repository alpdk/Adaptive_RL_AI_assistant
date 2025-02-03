import os
import sys
import torch
import argparse

from tqdm import tqdm
import numpy as np

from games.Game import Game
from models_structures.ResNet import ResNet
from src.games.Checkers import Checkers


def parse_arguments():
    """
    Parse command line arguments

    Returns:
         args (Namespace): parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for Compression of models')

    parser.add_argument('game', type=str,
                        help='Game that will be played between models')

    parser.add_argument('rounds_number', type=int,
                        help='Amount of rounds to play')

    parser.add_argument('local_algorithm_name', type=str,
                        help='Name of local trained model with specific algorithm weights')

    parser.add_argument('local_model_structure', type=str,
                        help='Local model structure that will be used')

    parser.add_argument('external_algorithm_name', type=str,
                        help='Name of external trained model with specific algorithm weights')

    parser.add_argument('external_model_structure', type=str,
                        help='External model structure that will be used')

    return parser.parse_args()


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
            return Checkers(8, 8)
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
    model.load_state_dict(torch.load(path, weights_only=True))

    return model


def main():
    """
    Main function for Compression of models
    """
    args = parse_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    game = load_game(args.game)

    if game is None:
        print("Game does not exist!!!")
        return 1

    model1 = load_model(device, game, "models_weights", args.local_algorithm_name, args.local_model_structure)
    model2 = load_model(device, game, "external_weights", args.external_algorithm_name, args.external_model_structure)

    if model1 is None or model2 is None:
        print("Games cannot be played without both players!!!")
        return 1

    cur_player_model = 1
    cur_player = 1

    score = {"win": 0, "loss": 0, "draw": 0}

    for i in tqdm(range(args.rounds_number)):
        state = game.get_initial_state()

        while True:
            valid_moves = game.get_valid_moves(state, cur_player)

            if cur_player_model == 1:
                policy, _ = model1(torch.tensor(game.get_encoded_state(state), device=model1.device).unsqueeze(0))
            else:
                policy, _ = model2(torch.tensor(game.get_encoded_state(state), device=model1.device).unsqueeze(0))

            policy = torch.softmax(policy, axis=1).squeeze(0).cpu().detach().numpy()

            valid_moves_list = np.zeros(game.action_size, dtype=bool)
            valid_moves_list[valid_moves] = True

            policy = policy * valid_moves_list
            action = np.argmax(policy)
            played_action = game.index_to_move[action]

            state = game.get_next_state(state, action, cur_player)

            value, is_terminal = game.get_value_and_terminated(state, cur_player)

            if is_terminal:
                print(state)

                if value == 1:
                    if cur_player_model == 1:
                        score["win"] += 1
                    else:
                        score["loss"] += 1
                else:
                    print("draw")
                break

            cur_player = game.get_next_player(state, action, cur_player)

            if cur_player == 1:
                cur_player_model = 1
            else:
                cur_player_model = 2

    print(score)


if __name__ == '__main__':
    main()
