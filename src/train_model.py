import torch
import argparse

from src.rl_algorithms.AlphaZero.AlphaZero import AlphaZero

from src.models_compression import load_model_structure, load_game

train_args = {
    'C': 2,
    'num_iterations': 1,
    'num_searches': 200,
    'num_selfPlay_iterations': 200,
    'num_epochs': 50,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

def parse_arguments():
    """
    Parse command line arguments

    Returns:
         args (Namespace): parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for training a model')

    parser.add_argument('game', type=str,
                        help='Game that will be played between models')

    parser.add_argument('model_structure_name', type=str,
                        help='Local model structure that will be used')

    parser.add_argument('rj_algorithm_name', type=str,
                        help='Name of local trained model with specific algorithm weights')

    return parser.parse_args()

def load_rl_algo(model, optimizer, game, rl_algorithm_name):
    rl_algorithm = rl_algorithm_name.lower()

    match rl_algorithm:
        case 'alphazero':
            return AlphaZero(model, optimizer, game, train_args)
        case _:
            return None

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

    model = load_model_structure(args.model_structure_name, game, device)

    if model is None:
        print("Cannot work without a model!!!")
        return 1

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    rl_algo = load_rl_algo(model, optimizer, game, args.rj_algorithm_name)

    if rl_algo is None:
        print(f"There is no such algorithm: {args.rj_algorithm_name}!!!")
        return 1

    rl_algo.learn()


if __name__ == '__main__':
    main()