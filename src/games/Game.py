import numpy as np

from abc import abstractmethod

class Game:
    """
    This is a parent class, for games, that will be used for training model

    Attributes:
        row_count (int): count of rows in the game
        column_count (int): count of columns in the game
        figures_kinds ([int]): a list of possible figure kinds in integer format
                               (same figures for different players should have different numbers)
        index_to_move ({int -> ...}): dictionary mapping index to move (move can be of any type)
        move_to_index ({... -> int}): dictionary mapping move to index (move can be of any type)
        action_size (int): number of possible actions
    """
    row_count = None
    column_count = None
    figures_kinds = None
    index_to_move = None
    move_to_index = None
    action_size = None

    def get_moves_to_np_array(self, valid_moves):
        """
        Returns a numpy array with a size equal to the number of actions.
        If move is valid make it equal to 1, otherwise 0.

        Args:
            valid_moves (np.array(int)): list of valid moves in indexes

        Returns:
            np.array(): array with a length of all actions
        """
        res = np.zeros(self.action_size, dtype=int)
        res[valid_moves] = 1
        return res

    def get_encoded_state(self, state):
        """
        Returns the encoded state of the game in a format of boards, where every board contain
        only 1 type of the figures.

        Args:
            state (np.array): 2d array of shape (rows, columns)

        Returns:
            np.array(): 3d array of shape (len(figures_kinds), rows, columns)
        """
        encoded_state = np.stack(
            [state == condition for condition in self.figures_kinds]
        ).astype(np.float32)  # 2 represents the kinged pieces

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    @abstractmethod
    def _get_figures_kinds(self):
        """
        Returns a list of possible figure kinds in integer format.
        You should create your own bijection between numbers and figures.
        Empty spot also classified as a figure.

        Returns:
            [int]: List of possible figures in integer format
        """
        pass

    @abstractmethod
    def _get_index_to_move(self):
        """
        Create a dictionary mapping index to move (move can be of any type)

        Returns:
            {}: Dictionary mapping index to move (move can be of any type)
        """
        pass

    @abstractmethod
    def get_initial_state(self):
        """
        Returns the initial state of the game's board

        Returns:
            np.array(): 2d array of shape (rows, columns) with initial state of figures
        """
        pass

    @abstractmethod
    def get_next_state(self, state, action, player):
        """
        Generate a new state that will be reached, after making an action by players at current board

        Args:
            state (np.array): 2d array of shape (rows, columns)
            action (int): the index of action to take
            player (int): the index of the player who takes the action

        Returns:
            np.array(): 2d array of shape (rows, columns)
        """
        pass

    @abstractmethod
    def get_valid_moves(self, state, cur_player=1):
        """
        Returns a list of valid moves that can be executed by a player (moves written in index format)

        Args:
            state (np.array): 2d array of shape (rows, columns)
            cur_player (int): the index of the current player

        Returns:
            valid_moves (np.array(int)): list of valid moves in indexes
        """
        pass

    @abstractmethod
    def get_next_player(self, state, action, player):
        """
        Returns the next player that will take the action.
        On the base of the last move.

        Args:
            state (np.array): current game state
            action (int): the index of the last taken action
            player (int): the index of the player who took the action

        Returns:
            (int): index of the next player
        """
        pass

    @abstractmethod
    def get_value_and_terminated(self, state, player):
        """
        Returns current value of the game and terminated or not is it.

        Args:
            state (np.array): current game state
            player (int): the index of the player who took the action

        Returns:
            value (int): value of the game
            terminated (bool): terminated or not
        """
        pass
