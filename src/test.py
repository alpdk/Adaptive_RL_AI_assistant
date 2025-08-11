# import os
#
# import torch
# import numpy as np
#
# from tqdm import tqdm
# from src.models_structures.base_play.ResNet import ResNet
# from src.games.UltimateTicTacToe import UltimateTicTacToe
#
# game = UltimateTicTacToe(3, 3)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = ResNet(game, 9, 128, device)
#
# path = os.path.dirname(os.path.abspath(__file__))
# path = path + f'/model_weights/base/UltimateTicTacToe.pt'
#
# model.load_state_dict(torch.load(path, map_location=device), strict=False)
#
# score = {"win": 0, "loss": 0, "draw": 0}
#
# for i in tqdm(range(10)):
#     cur_player = 1
#     current_balance = 0
#
#     while True:
#         state = game.logger.current_state
#         valid_moves = game.get_valid_moves(cur_player)
#
#         policy, _ = model(
#                 torch.tensor(game.get_encoded_state(game.logger.current_state), device=model.device).unsqueeze(0))
#
#         policy = game.get_normal_policy(policy, cur_player)
#         # moves_values = game.get_normal_values(moves_values, cur_player)
#
#         # game.logger.set_action_probs_and_values(policy, moves_values)
#
#         if np.isnan(policy).any():
#             print("Found NaNs in probabilities")
#
#         # action = np.random.choice(game.action_size, p=policy)
#         action = np.argmax(policy)
#         played_action = game.index_to_move[action]
#
#         # current_balance += cur_player * moves_values[action]
#
#         game.make_move(action, cur_player)
#
#         state = game.logger.current_state
#
#         value, is_terminal = game.get_value_and_terminated(cur_player)
#
#         if is_terminal:
#             state = game.logger.current_state
#             print("===============================================")
#             print(state)
#             print("===============================================")
#
#             if value == 1:
#                 if cur_player == 1:
#                     score["win"] += 1
#                 else:
#                     score["loss"] += 1
#             else:
#                 score["draw"] += 1
#
#             game.not_capture_moves = 0
#
#             break
#
#         cur_player = game.get_next_player(action, cur_player)
#
#     game.revert_full_game()
#
# print(score)
import torch
from nbformat.sign import algorithms

# from src.games.UltimateTicTacToe import UltimateTicTacToe
#
# game = UltimateTicTacToe()
#
# while True:
#     print(game.get_valid_moves())
#
#     move = int(input("What is your move: "))
#
#     game.make_move(move)
#
#     winner = game.get_value_and_terminated()
#
#     if winner != 0:
#         print(f"Our winner is {winner}!")
#         break
#
#     encoded_state = game.get_encoded_state()

# import torch
# import torch.nn as nn
#
# from src.models_structures.ResNet import ResNet
# from src.models_structures.ModifiedResNet import ModifiedResNet
# from src.games.UltimateTicTacToe import UltimateTicTacToe
#
# game = UltimateTicTacToe()
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # base_weights = torch.load('src/weights/model/model_9_UTTT.pt', map_location=device)
# model = ModifiedResNet(game, 9, 128, device=device)
#
# model.load_state_dict(torch.load('src/weights/model/modified_model_UTTT.pt', map_location=device))
# model.to(device)
# model.eval()  # if you're going to use it for inference
#
# encoded_state = torch.tensor(game.get_encoded_state(),device=model.device).unsqueeze(0)
# policy, move_values, value = model(encoded_state)
# policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
#
# print(policy)

from src.games.UltimateTicTacToe import UltimateTicTacToe
from src.models_structures.ModifiedResNet import ModifiedResNet
from src.rl_algorithms.MCTS import MCTS
from src.trainers.Trainer import Trainer

args = {
    'C':  0.57,
    'tau': 0.737,
    'num_iterations': 1,
    'num_searches': 100,
    'num_selfPlay_iterations': 128,
    'num_epochs': 100,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'history_depth': 5,
    'relevance_coef': 1.0
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

game = UltimateTicTacToe()
model = ModifiedResNet(game, 9, 128, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

algorithms = MCTS(args)
trainer = Trainer(args)

trainer.learn(game, model, optimizer, algorithms)
