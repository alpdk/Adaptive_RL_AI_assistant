import torch
import numpy as np

from .RLAlgorithms import RLAlgorithm
from src.rl_algorithms.Node import Node

class MCTS(RLAlgorithm):
    """
    Class implementing Monte-Carlo Tree Search (MCTS) algorithm.

    Attributes:
        game (Game): Game that will be played.
        args ({}): some arguments that will be passed to the MCTS algorithm.
        model(nn.Module): the model that will be trained.
    """
    def __init__(self, game, args, model):
        """
        Constructor.
        """
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state, player):
        """
        Method that performs MCTS search.

        Args:
            state (np.array): current state of the game
            player (int): current player

        Returns:
            action (np.array): action probabilities
        """
        root = Node(self.game, self.args, state, player, visit_count = 1)

        policy, value = self.model(torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

        policy = (1 - self.args['dirichlet_epsilon']) * policy + np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size) * self.args['dirichlet_epsilon']

        valid_moves = self.game.get_valid_moves(state, player)
        valid_moves = self.game.get_moves_to_np_array(valid_moves)
        policy = policy * valid_moves

        policy = policy / np.sum(policy)

        root.expand(policy, player)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            last_move_player = node.parent.player
            value, is_terminal = self.game.get_value_and_terminated(node.state, last_move_player)

            if not is_terminal:
                policy, value = self.model(torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
                cur_player = node.player
                valid_moves = self.game.get_valid_moves(node.state, cur_player)
                valid_moves = self.game.get_moves_to_np_array(valid_moves)
                policy = policy * valid_moves
                policy = policy / np.sum(policy)

                value = value.item()

                node.expand(policy, cur_player)

            node.backpropagate(value)


        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action = child.action_taken
            action_probs[action] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs