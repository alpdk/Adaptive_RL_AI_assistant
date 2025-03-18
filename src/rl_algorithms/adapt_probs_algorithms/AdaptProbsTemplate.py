from abc import abstractmethod

class AdaptProbsTemplate:
    """
    This is a template class for adaptation methods.

    Attributes:
        game (Game): game, that is played with history
        history_depth (int): how many steps, will be chosen for current one
        algorithm_name (str): name of the algorithm.
        extra_args ({}}): dictionary of additional args
    """

    def __init__(self, game, history_depth, algorithm_name, extra_args=None):
        """
        Constructor.

        Parameters:
            game (Game): game, that is played with history
            history_depth (int): how many steps, will be chosen for current one
            algorithm_name (str): name of the algorithm
            extra_args ({}}): dictionary of additional args
        """
        self.game = game
        self.history_depth = history_depth
        self.algorithm_name = algorithm_name
        self.extra_args = extra_args

    @abstractmethod
    def calc_player_income(self, opponent_moves):
        """
        Method for calculating opponent player's income for last n moves.

        Parameters:
             opponent_moves (np.ndarray): opponent moves

        Returns:
            int: median income by player
        """
        pass

    @abstractmethod
    def probs_modification(self):
        """
        Method for probs modification

        Returns:
            median_opponent_income (int): median income by player
            new_probs (np_array()): new probabilities
        """
        pass