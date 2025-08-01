import torch
import numpy as np

from src.rl_algorithms.base_rl_algorithms.BaseProbsTemplate import BaseProbsTemplate
from src.rl_algorithms.base_rl_algorithms.Node import Node


class MCTS(BaseProbsTemplate):
    """
    Class implementing Monte-Carlo Tree Search (MCTS) algorithm.

    Attributes:
        game (Game): Game that will be played.
        args ({}): some arguments that will be passed to the MCTS algorithm.
        model(nn.Module): the model that will be trained.
        algorithm_name (str): name of the algorithm.
    """

    def __init__(self, game, args, model):
        """
        Constructor.

        Parameters:
            game (Game): Game that will be played.
            args ({}): some arguments that will be passed to the RL algorithm.
            model (nn.Module): the model that will be trained.
        """
        super().__init__(game, args, model)
        self.algorithm_name = "MCTS"

    def search(self, player):
        """
        Method that performs MCTS search.

        Args:
            player (int): current player

        Returns:
            action_probs (np.array): action probabilities
            action_values (np.array): action values
        """
        root = Node(self.game, self.args, player, None, None, visit_count=1)
        # print(self.game)

        policy, _, value = self.model(torch.tensor(self.game.get_encoded_state(self.game.logger.current_state),
                                                   device=self.model.device).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

        policy = (1 - self.args['dirichlet_epsilon']) * policy + np.random.dirichlet(
            [self.args['dirichlet_alpha']] * self.game.action_size) * self.args['dirichlet_epsilon']

        valid_moves = self.game.get_valid_moves(player)
        valid_moves = self.game.get_moves_to_np_array(valid_moves)

        policy = policy * valid_moves
        policy = policy / np.sum(policy)

        root.expand(policy, player)

        # print("First move in search finished!")

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                if node.policy is None:
                    policy, _, _ = self.model(torch.tensor(self.game.get_encoded_state(self.game.logger.current_state),
                                                           device=self.model.device).unsqueeze(0))
                    cur_player = node.player

                    policy = self.game.get_normal_policy(policy, cur_player)

                    node.policy = policy

                node = node.select()
                self.game.make_move(node.action_taken, node.parent.player)

            last_move_player = node.parent.player
            value, is_terminal = self.game.get_value_and_terminated(last_move_player)

            if not is_terminal:
                # print('Terminal State reached')
                policy, _, value = self.model(torch.tensor(self.game.get_encoded_state(self.game.logger.current_state),
                                                           device=self.model.device).unsqueeze(0))
                cur_player = node.player

                policy = self.game.get_normal_policy(policy, cur_player)
                value = value.item()

                node.expand(policy, cur_player)

            node.backpropagation(value, last_move_player)

        action_values = self.calc_values(root)
        action_probs = self.calc_probs(root, action_values)
        # action_visits = self.calc_visits(root)

        # node = root
        #
        # while node.is_fully_expanded():
        #     node = node.select(player)
        #     self.game.make_move(node.action_taken, node.parent.player)
        # print('Finished search')

        return action_probs, action_values
