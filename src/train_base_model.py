import math

from src.external_methods_and_arguments import *

train_args = {
    'C': math.sqrt(2),
    'tau': 1,
    'num_iterations': 5,
    'num_searches': 100,
    'num_selfPlay_iterations': 200,
    'num_epochs': 100,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'history_depth': 5,
    'relevance_coef': 1.0
}

def parse_base_model_arguments():
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

    parser.add_argument('rl_algorithm_name', type=str,
                        help='Name of local trained model with specific algorithm weights')

    return parser.parse_args()

def main():
    """
    Main function for training base model
    """
    args = parse_base_model_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    game = load_game(args.game)

    if game is None:
        print("Game does not exist!!!")
        return 1

    model = load_model_structure(args.model_structure_name, game, device)

    if model is None:
        print("Cannot work without a model!!!")
        return 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    rl_algorithm = load_rl_algorithm(args.rl_algorithm_name, game, train_args, model)

    trainer = load_trainer("BaseTrainer", rl_algorithm, model, None, optimizer, game, train_args)

    if trainer is None:
        print(f"There is no such algorithm: {args.rl_algorithm_name}!!!")
        return 1

    trainer.learn()


if __name__ == '__main__':
    main()