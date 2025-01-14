# import math
# import numpy as np
#
# import torch
#
# import torch.nn as nn
# import torch.nn.functional as F
#
# from tqdm import trange
#
# import random
#
# torch.manual_seed(0)
#
# class Checkers:
#     def __init__(self):
#         self.row_count = 8
#         self.column_count = 8
#         self.action_size = 170
#         self.valid_squares = self._get_valid_squares()
#         self.index_move_to_tuple_move = self._get_index_move_to_tuple_move()
#         self.tuple_move_to_index_move = {v: k for k, v in self.index_move_to_tuple_move.items()}
#         # print(self.valid_squares)
#
#     def _get_index_move_to_tuple_move(self):
#         res = {}
#
#         directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]
#         index = 0
#
#         for row, col in self.valid_squares:
#             for dr, dc in directions:
#                 new_row = row + dr
#                 new_col = col + dc
#
#                 if 0 <= new_row < self.row_count and 0 <= new_col < self.column_count:
#                     start_pos = self.valid_squares.index((row, col)) + 1
#                     end_pos = self.valid_squares.index((new_row, new_col)) + 1
#
#                     res[index] = (start_pos, end_pos)
#                     index = index + 1
#
#         return res
#
#     def _get_valid_squares(self):
#         """Returns a list of valid squares (dark squares) on an 8x8 board"""
#         valid_squares = []
#         for row in range(self.row_count):
#             for col in range(self.column_count):
#                 # Dark squares are those where (row + col) is odd
#                 if (row + col) % 2 == 1:
#                     valid_squares.append((row, col))
#         return valid_squares
#
#     def get_initial_state(self):
#         # Initial state: set up pieces for each player
#         # Player 1 is represented by 1, and Player 2 by -1
#         state = np.zeros((self.row_count, self.column_count))
#         for i, (row, col) in enumerate(self.valid_squares[:12]):  # First 12 squares for Player 2
#             state[row, col] = -1
#         for i, (row, col) in enumerate(self.valid_squares[20:]):  # Last 12 squares for Player 1
#             state[row, col] = 1
#         return state
#
#     def get_next_state(self, state, action, player):
#         # Action: a tuple (start_index, end_index) in the range 1-32
#         start_index, end_index = action
#         start_row, start_col = self.valid_squares[start_index - 1]
#         end_row, end_col = self.valid_squares[end_index - 1]
#
#         new_state = state.copy()
#
#         # Move the piece
#         new_state[start_row, start_col], new_state[end_row, end_col] = new_state[end_row, end_col], new_state[start_row, start_col]
#
#         # Check if a piece needs to be "kinged"
#         if (player == 1 and end_row == 0) or (player == -1 and end_row == self.row_count - 1):
#             new_state[end_row, end_col] = player * 2  # Kinged piece
#
#         # Check for capture
#         if abs(start_row - end_row) == 2:
#             mid_row = (start_row + end_row) // 2
#             mid_col = (start_col + end_col) // 2
#             new_state[mid_row, mid_col] = 0  # Capture the opponent's piece
#
#         return new_state
#
#     def get_capture_moves(self, state, player):
#         res = []
#         directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
#
#         for i, (row, col) in enumerate(self.valid_squares):
#             if state[row, col] == player or state[row, col] == player * 2:
#                 for dr, dc in directions:
#                     if dr * player > 0 and state[row, col] != player * 2:
#                         continue
#
#                     new_row = row + dr
#                     new_col = col + dc
#
#                     if 0 <= new_row < self.row_count and 0 <= new_col < self.column_count:
#                         if state[new_row, new_col] == -player or state[new_row, new_col] == -player * 2:
#                             capture_row = new_row + dr
#                             capture_col = new_col + dc
#
#                             if 0 <= capture_row < self.row_count and 0 <= capture_col < self.column_count and state[capture_row, capture_col] == 0:
#                                 res.append((i + 1, self.valid_squares.index((capture_row, capture_col)) + 1))
#         return res
#
#     def get_normal_moves(self, state, player):
#         res = []
#         directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
#
#         for i, (row, col) in enumerate(self.valid_squares):
#             if state[row, col] == player or state[row, col] == player * 2:
#                 for dr, dc in directions:
#                     if dr * player > 0 and state[row, col] != player * 2:
#                         continue
#
#                     new_row = row + dr
#                     new_col = col + dc
#
#                     if 0 <= new_row < self.row_count and 0 <= new_col < self.column_count and state[new_row, new_col] == 0:
#                         res.append((i + 1, self.valid_squares.index((new_row, new_col)) + 1))
#         return res
#
#     def moves_to_np_arr(self, valid_moves):
#         res = np.zeros(self.action_size)
#         directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]
#         index = 0
#
#         for row, col in self.valid_squares:
#             for dr, dc in directions:
#                 new_row = row + dr
#                 new_col = col + dc
#
#                 if 0 <= new_row < self.row_count and 0 <= new_col < self.column_count:
#                     start_pos = self.valid_squares.index((row, col)) + 1
#                     end_pos = self.valid_squares.index((new_row, new_col)) + 1
#
#                     if (start_pos, end_pos) in valid_moves:
#                         res[index] = 1
#                     index = index + 1
#
#         return res
#
#     def get_valid_moves(self, state, cur_player=1):
#         player = cur_player
#         valid_moves = self.get_capture_moves(state, player)
#
#         if valid_moves == []:
#             valid_moves = self.get_normal_moves(state, player)
#
#         return valid_moves
#
#     def get_next_player(self, state, action, player):
#         next_player = self.get_opponent(player)
#
#         if abs(action[0] - action[1]) >= 7:
#             valid_moves = self.get_valid_moves(state, player)
#
#             for start, end in valid_moves:
#                 if start == action[1] and abs(start - end) >= 7:
#                     next_player = player
#                     break
#
#         return next_player
#
#     def check_win(self, state, player):
#         # A player wins if the opponent has no valid moves left
#         opponent = self.get_opponent(player)
#         return len(self.get_valid_moves(state, opponent)) == 0
#
#
#     def get_value_and_terminated(self, state, player):
#         if self.check_win(state, player):
#             return 1, True  # Player wins
#         return 0, False
#
#     def get_opponent(self, player):
#         return -player
#
#     def get_opponent_value(self, value):
#         return -value
#
#     def change_perspective(self, state, player):
#         return state * player
#
#     def get_move_from_index(self, index):
#         return self.index_move_to_tuple_move[index]
#
#     def get_encoded_state(self, state):
#         encoded_state = np.stack(
#             (state == -1, state == 0, state == 1, state == 2, state == -2)
#         ).astype(np.float32)  # 2 represents the kinged pieces
#
#         if len(state.shape) == 3:
#             encoded_state = np.swapaxes(encoded_state, 0, 1)
#
#         return encoded_state
#
# class ResNet(nn.Module):
#     def __init__(self, game, num_resBlocks, num_hidden, device):
#         super().__init__()
#
#         self.device = device
#         self.startBlock = nn.Sequential(
#             nn.Conv2d(5, num_hidden, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU()
#         )
#
#         self.backBone = nn.ModuleList(
#             [ResBlock(num_hidden) for i in range(num_resBlocks)]
#         )
#
#         self.policyHead = nn.Sequential(
#             nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(32 * game.row_count * game.column_count, game.action_size)
#         )
#
#         self.valueHead = nn.Sequential(
#             nn.Conv2d(num_hidden, 5, kernel_size=3, padding=1),
#             nn.BatchNorm2d(5),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(5 * game.row_count * game.column_count, 1),
#             nn.Tanh()
#         )
#
#         self.to(device)
#
#     def forward(self, x):
#         # print(x.shape)
#         x = self.startBlock(x)
#         for resBlock in self.backBone:
#             x = resBlock(x)
#         policy = self.policyHead(x)
#         value = self.valueHead(x)
#         return policy, value
#
#
# class ResBlock(nn.Module):
#     def __init__(self, num_hidden):
#         super().__init__()
#         self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(num_hidden)
#         self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(num_hidden)
#
#     def forward(self, x):
#         residual = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += residual
#         x = F.relu(x)
#         return x
#
# class Node:
#     def __init__(self, game, args, state, player = 0, parent=None, action_taken=None, prior=0, visit_count=0):
#         self.game = game
#         self.args = args
#         self.state = state
#         self.player = player
#         self.parent = parent
#         self.action_taken = action_taken
#         self.prior = prior
#
#         self.children = []
#
#         self.visit_count = visit_count
#         self.value_sum = 0
#
#     def is_fully_expanded(self):
#         return len(self.children) > 0
#
#     def select(self):
#         best_child = None
#         best_ucb = -np.inf
#
#         for child in self.children:
#             ucb = self.get_ucb(child)
#             if ucb > best_ucb:
#                 best_child = child
#                 best_ucb = ucb
#
#         return best_child
#
#     def get_ucb(self, child):
#         if child.visit_count == 0:
#             q_value = 0
#         else:
#             q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
#         return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
#
#     def expand(self, policy, player):
#         for action, prob in enumerate(policy):
#             if prob > 0:
#                 child_state = self.state.copy()
#                 action = self.game.get_move_from_index(action)
#                 child_state = self.game.get_next_state(child_state, action, player)
#
#                 next_player = self.game.get_next_player(child_state, action, player)
#
#                 child = Node(self.game, self.args, child_state, next_player, self, action, prob)
#                 self.children.append(child)
#
#         # return child
#
#     def backpropagate(self, value):
#         self.value_sum += value
#         self.visit_count += 1
#
#         value = self.game.get_opponent_value(value)
#         if self.parent is not None:
#             self.parent.backpropagate(value)
#
# class MCTS:
#     def __init__(self, game, args, model):
#         self.game = game
#         self.args = args
#         self.model = model
#
#     @torch.no_grad()
#     def search(self, state, player):
#         root = Node(self.game, self.args, state, player, visit_count = 1)
#
#         policy, value = self.model(torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0))
#         policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
#
#         policy = (1 - self.args['dirichlet_epsilon']) * policy + np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size) * self.args['dirichlet_epsilon']
#
#         valid_moves = self.game.get_valid_moves(state, player)
#         valid_moves = self.game.moves_to_np_arr(valid_moves)
#         policy = policy * valid_moves
#
#         policy = policy / np.sum(policy)
#
#         root.expand(policy, player)
#
#         for search in range(self.args['num_searches']):
#             node = root
#
#             while node.is_fully_expanded():
#                 node = node.select()
#
#             last_move_player = node.parent.player
#             value, is_terminal = self.game.get_value_and_terminated(node.state, last_move_player)
#             value = self.game.get_opponent_value(value)
#
#             if not is_terminal:
#                 policy, value = self.model(torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0))
#                 policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
#                 cur_player = node.player
#                 valid_moves = self.game.get_valid_moves(node.state, cur_player)
#                 valid_moves = self.game.moves_to_np_arr(valid_moves)
#                 policy = policy * valid_moves
#                 policy = policy / np.sum(policy)
#
#                 value = value.item()
#
#                 node.expand(policy, cur_player)
#
#             node.backpropagate(value)
#
#
#         action_probs = np.zeros(self.game.action_size)
#         for child in root.children:
#             action = self.game.tuple_move_to_index_move[child.action_taken]
#             action_probs[action] = child.visit_count
#         action_probs /= np.sum(action_probs)
#         return action_probs
#
# class AlphaZero:
#     def __init__(self, model, optimizer, game, args):
#         self.model = model
#         self.optimizer = optimizer
#         self.game = game
#         self.args = args
#         self.mcts = MCTS(game, args, model)
#
#     def selfPlay(self):
#         memory = []
#         player = 1
#         state = self.game.get_initial_state()
#
#         while True:
#             neutral_state = state.copy()
#             action_probs = self.mcts.search(neutral_state, player)
#
#             memory.append((neutral_state, action_probs, player))
#
#             temperature_action_probs = action_probs ** (1 / self.args['temperature'])
#             # action_probs = temperature_action_probs / np.sum(temperature_action_probs)
#             action_index = np.random.choice(self.game.action_size, p=action_probs)
#             action = self.game.get_move_from_index(action_index)
#             state = self.game.get_next_state(state, action, player)
#
#             value, is_terminal = self.game.get_value_and_terminated(state, player)
#
#             if is_terminal:
#                 returnMemory = []
#                 for hist_neutral_state, hist_action_probs, hist_player in memory:
#                     hist_outcome = value if hist_player == player else -value
#
#                     returnMemory.append((
#                         self.game.get_encoded_state(hist_neutral_state),
#                         hist_action_probs,
#                         hist_outcome
#                     ))
#                 return returnMemory
#
#             player = self.game.get_next_player(state, action, player)
#
#     def train(self, memory):
#         random.shuffle(memory)
#
#         for batchIdx in range(0, len(memory), self.args['batch_size']):
#             sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
#             state, policy_targets, value_targets = zip(*sample)
#
#             state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets)
#
#             state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
#             policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
#             value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
#
#             out_policy, out_value = self.model(state)
#
#             policy_loss = F.cross_entropy(out_policy, policy_targets)
#             value_loss = F.mse_loss(out_value, value_targets)
#
#             loss = policy_loss + value_loss
#
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#     def learn(self):
#         for iteration in trange(self.args['num_iterations']):
#             memory = []
#
#             self.model.eval()
#
#             for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
#                 memory += self.selfPlay()
#
#             self.model.train()
#
#             for epoch in trange(self.args['num_epochs']):
#                 self.train(memory)
#
#             torch.save(self.model.state_dict(), f"model_{iteration}.pth")
#             torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
#
# checkers = Checkers()
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# print(device)
#
# model = ResNet(checkers, 4, 64, device)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#
# args = {
#     'C': 2,
#     'num_searches': 60,
#     'num_iterations': 8,
#     'num_selfPlay_iterations': 50,
#     'num_epochs': 4,
#     'batch_size': 64,
#     'temperature': 1.25,
#     'dirichlet_epsilon': 0.25,
#     'dirichlet_alpha': 0.3
# }
#
# alphaZero = AlphaZero(model, optimizer, checkers, args)
# alphaZero.learn()

import torch

from games.Checkers import Checkers
from AlphaZero.ResNet import ResNet
from AlphaZero.AlphaZero import AlphaZero

checkers = Checkers()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = ResNet(checkers, 4, 64, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 8,
    'num_selfPlay_iterations': 1,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alphaZero = AlphaZero(model, optimizer, checkers, args)
alphaZero.learn()