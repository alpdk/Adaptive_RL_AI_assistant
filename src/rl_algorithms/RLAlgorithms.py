from abc import abstractmethod

class RLAlgorithm:
    """
    This is a parent class, for algorithms, that will be used for training model

    Attributes:
        game  (Game): Game, that will be used in training
        args  ({}): Dictionary of arguments, that will be passed to the game
        model (nn.Module): The model that will be trained.
    """

    @abstractmethod
    def search(self, state, player):
        """
        Returns a list of probabilities for each action

        Args:
            state (np.array): current state of the game
            player (int): current player

        Returns:
            action (np.array): action probabilities
        """
        pass