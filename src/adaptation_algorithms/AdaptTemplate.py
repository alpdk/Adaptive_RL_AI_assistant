

class AdaptTemplate:
    """
    Mother class for adaptation algorithm

    Attributes:
        args (dict): dictionary of arguments
    """

    def __init__(self, args):
        """
        Constructor for AdaptTemplate

        Args:
            args (dict): dictionary of arguments
        """
        self.args = args

    def _history_collection(self, game):
        """
        Method for collecting the history of the game

        Args:
            game (Game): game from which we want to collect the history

        Returns:
            history (list): list of pointers to the last N changes
        """
        logger_pointer = game.logger
        history = []

        for i in range(self.args['history_depth']):
            if logger_pointer.parent != None:
                history.append(logger_pointer)
                logger_pointer = logger_pointer.parent
            else:
                break

        return history

    def _calc_balance(self, game, history):
        """
        Method for calculating the balance of the game

        Args:
            game (Game): game that we play
            history (list): history of the game

        Returns:
            balance (float): balance of the game
        """

    def update_probs(self, game, probs, values):
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
        pass