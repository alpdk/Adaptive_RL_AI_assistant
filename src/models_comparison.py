import torch
from tqdm import tqdm
import numpy as np

from src.external_methods_and_arguments import *

train_args = {
    'C': 2,
    'num_iterations': 5,
    'num_searches': 200,
    'num_selfPlay_iterations': 200,
    'num_epochs': 200,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'history_depth': 5,
    'relevance_coef': 1.0
}

def parse_comparison_arguments():
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

    parser.add_argument('local_base_algorithm_name', type=str,
                        help='Name of local trained base model with specific algorithm weights')

    parser.add_argument('local_base_model_structure', type=str,
                        help='Local base model structure that will be used')

    parser.add_argument('external_base_algorithm_name', type=str,
                        help='Name of external base trained model with specific algorithm weights')

    parser.add_argument('external_base_model_structure', type=str,
                        help='External base model structure that will be used')

    parser.add_argument('include_adapt_models', type=int,
                        help='Add adaptive models, or not')

    parser.add_argument('--adaptive_algorithm', type=str,
                        help='Name of adaptive algorithm')

    parser.add_argument('--local_adapt_algorithm_name', type=str,
                        help='Name of local trained adapt model with specific algorithm weights')

    parser.add_argument('--local_adapt_model_structure', type=str,
                        help='Local adapt model structure that will be used')

    parser.add_argument('--external_adapt_algorithm_name', type=str,
                        help='Name of external trained adapt model with specific algorithm weights')

    parser.add_argument('--external_adapt_model_structure', type=str,
                        help='External adapt model structure that will be used')

    return parser.parse_args()


def main():
    """
    Main function for Compression of models
    """
    args = parse_comparison_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    game = load_game(args.game)

    if game is None:
        print("Game does not exist!!!")
        return 1

    base_model1 = load_model(device, game, "base_models_weights", args.local_base_algorithm_name, args.local_base_model_structure)
    base_model2 = load_model(device, game, "external_base_weights", args.external_base_algorithm_name, args.external_base_model_structure)

    if base_model1 is None or base_model2 is None:
        print("There is no one of the such base models!!!")
        return 1

    base_model1.eval()
    base_model2.eval()

    if args.include_adapt_models:
        adapt_model_1 = load_model(device, game, "adapt_models_weights", args.local_adapt_algorithm_name, args.local_adapt_model_structure)
        adapt_model_2 = load_model(device, game, "external_adapt_weights", args.external_adapt_algorithm_name, args.external_adapt_model_structure)
        adapt_probs = load_adapt_algortihm(args.adaptive_algorithm, game, train_args['history_depth'], train_args['relevance_coef'])

        if adapt_model_1 is None or adapt_model_2 is None:
            print("There is no one of the such adaptive models!!!")
            return 1

        adapt_model_1.eval()
        adapt_model_2.eval()

    score = {"win": 0, "loss": 0, "draw": 0}

    for i in tqdm(range(args.rounds_number)):
        cur_player = 1
        current_balance = 0

        while True:
            state = game.logger.current_state
            valid_moves = game.get_valid_moves(cur_player)

            if cur_player == 1:
                policy, moves_values, _ = base_model1(
                    torch.tensor(game.get_encoded_state(game.logger.current_state), device=base_model1.device).unsqueeze(0))
            else:
                policy, moves_values, _ = base_model2(
                    torch.tensor(game.get_encoded_state(game.logger.current_state), device=base_model2.device).unsqueeze(0))

            policy = game.get_normal_policy(policy, cur_player)
            moves_values = game.get_normal_values(moves_values, cur_player)

            game.logger.set_action_probs_and_values(policy, moves_values)

            if args.include_adapt_models:
                opponent_moves = game.collect_opponent_moves(train_args['history_depth'])

                if cur_player == 1:
                    median_opponent_income = torch.tensor([adapt_probs.calc_player_income(opponent_moves)],
                                                          device=adapt_model_1.device)
                    # median_opponent_income = torch.tensor([1.0], device=adapt_model_2.device)

                    input = np.concatenate((median_opponent_income, torch.from_numpy(policy)), axis=0)

                    policy = adapt_model_1(torch.tensor(input, dtype=torch.float32, device=adapt_model_1.device).unsqueeze(0))
                else:
                    median_opponent_income = torch.tensor([adapt_probs.calc_player_income(opponent_moves)],
                                                          device=adapt_model_2.device)
                    # median_opponent_income = torch.tensor([1.0], device=adapt_model_2.device)

                    input = np.concatenate((median_opponent_income, torch.from_numpy(policy)), axis=0)

                    policy = adapt_model_2(torch.tensor(input, dtype=torch.float32, device=adapt_model_2.device).unsqueeze(0))

                policy = game.get_normal_policy(policy, cur_player)

            if np.isnan(policy).any():
                print("Found NaNs in probabilities")

            # action = np.random.choice(game.action_size, p=policy)
            action = np.argmax(policy)
            played_action = game.index_to_move[action]

            current_balance += cur_player * moves_values[action]

            game.make_move(action, cur_player)

            state = game.logger.current_state

            value, is_terminal = game.get_value_and_terminated(cur_player)

            if is_terminal:
                state = game.logger.current_state
                print("===============================================")
                print(state)
                print("===============================================")

                if value == 1:
                    if cur_player == 1:
                        score["win"] += 1
                    else:
                        score["loss"] += 1
                else:
                    score["draw"] += 1

                game.not_capture_moves = 0

                break

            cur_player = game.get_next_player(action, cur_player)

        game.revert_full_game()

    print(score)


if __name__ == '__main__':
    main()
