import torch
import numpy as np

from src.rl_algorithms.Node import Node

class MCTS:
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

    def calc_probs(self, root):
        """
        Method for calculating the probabilities of each action

        Parameters:
             root(Node): root of the current game

        Returns:
            np.array[]: probabilities of each action
        """
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action = child.action_taken
            action_probs[action] = child.visit_count
        action_probs /= np.sum(action_probs)

        return action_probs

    # def calc_values(self, root):
    #     """
    #     Method for calculating the values of each action
    #
    #     Parameters:
    #          root(Node): root of the current game
    #
    #     Returns:
    #         np.array[]: values of each action
    #     """
    #     action_values = np.zeros(self.game.action_size)
    #     for child in root.children:
    #         action = child.action_taken
    #         action_values[action] = child.value_sum
    #
    #     return action_values
    #
    # def calc_visits(self, root):
    #     """
    #     Method for calculating the amount visits of each action
    #
    #     Parameters:
    #          root(Node): root of the current game
    #
    #     Returns:
    #         np.array[]: visits of each action
    #     """
    #     action_visits = np.zeros(self.game.action_size)
    #     for child in root.children:
    #         action = child.action_taken
    #         action_visits[action] = child.visit_count
    #
    #     return action_visits

    @torch.no_grad()
    def search(self, player):
        """
        Method that performs MCTS search.

        Args:
            player (int): current player

        Returns:
            action (np.array): action probabilities
        """
        root = Node(self.game, self.args, player, None, None, visit_count = 1)

        policy, value = self.model(torch.tensor(self.game.get_encoded_state(self.game.logger.current_state), device=self.model.device).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

        policy = (1 - self.args['dirichlet_epsilon']) * policy + np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size) * self.args['dirichlet_epsilon']

        valid_moves = self.game.get_valid_moves(player)
        valid_moves = self.game.get_moves_to_np_array(valid_moves)
        policy = policy * valid_moves

        policy = policy / np.sum(policy)

        root.expand(policy, player)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select(player)
                self.game.make_move(node.action_taken, node.parent.player)

            last_move_player = node.parent.player
            value, is_terminal = self.game.get_value_and_terminated(last_move_player)

            if not is_terminal:
                policy, value = self.model(torch.tensor(self.game.get_encoded_state(self.game.logger.current_state), device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
                cur_player = node.player
                valid_moves = self.game.get_valid_moves(cur_player)
                valid_moves = self.game.get_moves_to_np_array(valid_moves)
                policy = policy * valid_moves
                policy = policy / np.sum(policy)

                value = value.item()

                node.expand(policy, cur_player)

            if player != last_move_player:
                value = -value

            node.backpropagate(value)

        action_probs = self.calc_probs(root)
        # action_values = self.calc_values(root)
        # action_visits = self.calc_visits(root)

        return action_probs