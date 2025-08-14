import torch


class AdaptiveRL:
    """
    Class to implement Monte Carlo Tree Search (MCTS)

    Attributes:
        args ({}): Some arguments, that will be passed used
        root (TreeNode): Root node of the MCTS tree
        current_node (TreeNode): Current node that will be used
        algorithm_name (str): Name of the algorithm
    """

    def __init__(self, args):
        """
        Constructor

        Args:
            args ({}): Some arguments, that will be passed used
            algorithm_name (str): Name of the algorithm
        """
        self.args = args
        # self.root = TreeNode(args)
        # self.current_node = self.root
        self.algorithm_name = "AdaptiveRL"

    def adaptive_probs(self, game, model, adapt_algorithm):
        """
        Method that performs MCTS search.

        Args:
            game (Game): Game that is used
            model (nn.Module): Model that will be used
            adapt_algorithm (AdaptTemplate): Algorithm for probabilities adaptation

        Returns:
            new_policy (np.array): new action probabilities
            target_value (np.float): target value for high policy actions
        """
        policy, values, _ = model(torch.tensor(game.get_encoded_state(game.state),
                                                       device=model.device).unsqueeze(0))

        policy = policy.squeeze(0).detach().cpu().numpy()
        values = values.squeeze(0).detach().cpu().numpy()

        valid_moves = game.get_valid_moves()
        valid_moves = game.get_moves_to_np_array(valid_moves)

        policy = policy * valid_moves
        values = values * valid_moves

        policy = game.get_normal_policy(policy)

        new_policy, target_value = adapt_algorithm.update_probs(game, policy, values)

        return new_policy, target_value

    # def calc_probs(self, game, values=None):
    #     """
    #     Method for calculating the probabilities of each action
    #
    #     Parameters:
    #         game (Game): Game that is used
    #         values (np.array[float]): array of values
    #
    #     Returns:
    #         np.array[]: probabilities of each action
    #     """
    #     pass
    #
    # def calc_values(self, game):
    #     """
    #     Method for calculating the values of each action
    #
    #     Parameters:
    #          game (Game): Game that is used
    #
    #     Returns:
    #         np.array[]: values of each action
    #     """
    #     pass
    #
    #
    # def algorithm_step(self, action_taken):
    #     """
    #     Method that makes a step in algorithm
    #
    #     Args:
    #         action_taken (index): Move index tha twas executed
    #     """
    #     pass
    #
    # def return_to_base(self):
    #     """
    #     Method for returning algorithm to the base condition
    #     """
    #     pass