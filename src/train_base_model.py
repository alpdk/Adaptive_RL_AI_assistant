import math

from src.external_methods_and_arguments import *

train_args = {
    'C':  0.57,
    'tau': 0.737,
    'num_iterations': 10,
    'num_searches': 500,
    'num_selfPlay_iterations': 256,
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

    parser.add_argument('load_weights', type=int,
                        help='Load wights or not')

    return parser.parse_args()

def load_weights(model, game, args):
    model_weights = "model_" + game.game_name.lower() + "_" + args.rl_algorithm_name.lower() + "_" + args.model_structure_name.lower() + ".pth"

    path = os.path.dirname(os.path.abspath(__file__))
    path = path + f'/base_models_weights/{model_weights}'

    if os.path.exists(path):
        print(
            f'Model trained for game \"{game.game_name.lower()}\", algorithm \"{args.rl_algorithm_name.lower()}\", and model structure \"{args.model_structure_name.lower()}\" does exist inside of the \"base_models_weights/\" directory!')
        model.load_state_dict(torch.load(path, weights_only=True))

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

    if args.load_weights == 1:
        load_weights(model, game, args)

    # model.load_state_dict(torch.load("/home/alpdk/gitRepos/Adaptive_RL_AI_assistant/src/base_models_weights/model_ultimatetictactoe_mcts_resnet.pth", weights_only=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer = model.get_optimizer()

    rl_algorithm = load_rl_algorithm(args.rl_algorithm_name, game, train_args, model)

    trainer = load_trainer("BaseTrainer", rl_algorithm, model, None, optimizer, game, train_args)

    if trainer is None:
        print(f"There is no such algorithm: {args.rl_algorithm_name}!!!")
        return 1

    trainer.learn()


if __name__ == '__main__':
    main()