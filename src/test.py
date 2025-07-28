import os

import torch
import numpy as np

from tqdm import tqdm
from src.models_structures.base_play.ResNet import ResNet
from src.games.UltimateTicTacToe import UltimateTicTacToe

game = UltimateTicTacToe(3, 3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet(game, 9, 128, device)

path = os.path.dirname(os.path.abspath(__file__))
path = path + f'/model_weights/base/UltimateTicTacToe.pt'

model.load_state_dict(torch.load(path, map_location=device), strict=False)

score = {"win": 0, "loss": 0, "draw": 0}

for i in tqdm(range(10)):
    cur_player = 1
    current_balance = 0

    while True:
        state = game.logger.current_state
        valid_moves = game.get_valid_moves(cur_player)

        policy, _ = model(
                torch.tensor(game.get_encoded_state(game.logger.current_state), device=model.device).unsqueeze(0))

        policy = game.get_normal_policy(policy, cur_player)
        # moves_values = game.get_normal_values(moves_values, cur_player)

        # game.logger.set_action_probs_and_values(policy, moves_values)

        if np.isnan(policy).any():
            print("Found NaNs in probabilities")

        # action = np.random.choice(game.action_size, p=policy)
        action = np.argmax(policy)
        played_action = game.index_to_move[action]

        # current_balance += cur_player * moves_values[action]

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
