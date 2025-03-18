from src.external_methods_and_arguments import *

train_args = {
    'C': 2,
    'num_iterations': 10,
    'num_searches': 500,
    'num_selfPlay_iterations': 500,
    'num_epochs': 500,
    'batch_size': 100,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'history_depth': 5,
    'relevance_coef': 1.0
}

def parse_adapt_model_arguments():
    """
    Parse command line arguments

    Returns:
         args (Namespace): parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for training adaptive layer of a model')

    parser.add_argument('game', type=str,
                        help='Game that will be played between models')

    parser.add_argument('base_model_structure_name', type=str,
                        help='Local model structure that will be used for base move probabilities')

    parser.add_argument('adaptive_model_structure_name', type=str,
                        help='Local model structure that will be used for base move probabilities')

    parser.add_argument('rl_algorithm_name', type=str,
                        help='Name of local trained model with specific algorithm weights')

    parser.add_argument('adaptive_algorithm_name', type=str,
                        help='Algorithm that adapt move probabilities on the base of opponent moves\' history')

    parser.add_argument('history_depth', type=int,
                        help='Amount of moves, that will be used for calculation of an opponent level')

    parser.add_argument('relevance_coef', type=float,
                        help='Coefficient that show how relevant move for calculation of opponent level')

    return parser.parse_args()

def main():
    """
    Main function for training adaptive layers of model
    """
    args = parse_adapt_model_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    game = load_game(args.game)

    if game is None:
        print("Game does not exist!!!")
        return 1

    base_model = load_model(device, game, "base_models_weights", args.rl_algorithm_name, args.base_model_structure_name)
    adaptive_model = load_model_structure(args.adaptive_model_structure_name, game, device)

    if base_model is None or adaptive_model is None:
        print("At least one of the models does not exists!!!")
        return 1

    optimizer = torch.optim.AdamW(adaptive_model.parameters(), lr=0.001, weight_decay=0.0001)

    algorithm = load_adapt_algortihm(args.adaptive_algorithm_name, game, args.history_depth, args.relevance_coef)

    trainer = load_trainer("AdaptTrainer", algorithm, base_model, adaptive_model, optimizer, game, train_args)

    trainer.learn()

if __name__ == '__main__':
    main()