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
        game_name (str): name of the game
        logger (LoggerNode): logger node
    """
    row_count = None
    column_count = None
    figures_kinds = None
    index_to_move = None
    move_to_index = None
    action_size = None
    game_name = None
    logger = None

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

        Returns:
            np.array(): 3d array of shape (len(figures_kinds), rows, columns)
        """
        encoded_state = np.stack(
            [state == condition for condition in self.figures_kinds]
        ).astype(np.float32)  # 2 represents the kinged pieces

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    def collect_opponent_moves(self, history_depth):
        """
        Method for collecting opponent moves.

        Args:
            history_depth (int): how many opponent moves to collect

        Returns:
             np.ndarray: opponent moves
        """
        current_player = self.logger.current_player

        opponent_moves = np.empty(shape=(history_depth,), dtype=object)
        logger_ref = self.logger

        index = 0
        logger_ref = logger_ref.parent

        while index < self.history_depth and logger_ref is not None:
            if current_player != logger_ref.current_player:
                opponent_moves[index] = logger_ref
                index += 1

            logger_ref = logger_ref.parent

        return opponent_moves

    def get_some_history(self, history_depth):
        """
        Method for collecting input for model

        Args:
            history_depth (int): how many opponent moves to collect

        Returns:
            np_array(): encoded states hor last history_depth steps of an opponent
        """

        opponent_moves = self.collect_opponent_moves()

        new_input = np.empty(shape=(history_depth + 1, ), dtype=object)

        new_input[0] = self.get_encoded_state(self.logger.current_state)

        for i in range(history_depth):
            if opponent_moves[i] is None:
                new_input[i + 1] = self.get_encoded_state(np.zeros(shape=self.logger.current_state.shape, dtype=int))
            else:
                new_input[i + 1] = self.get_encoded_state(opponent_moves[i].current_state)

        res = np.stack(new_input)
        res = res.reshape(-1, self.row_count, self.column_count)
        return res


    @abstractmethod
    def _get_figures_kinds(self):
        """
        Returns a list of possible figure kinds in integer format.
        You should create your own bijection between numbers and figures.
        Empty spot also classified as a figure.

        Returns:
            [int]: List of possible figures in integer format.
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
    def _get_initial_state(self):
        """
        Returns the initial state of the game's board.

        Returns:
            np.array(): 2d array of shape (rows, columns) with initial state of figures.
        """
        pass

    @abstractmethod
    def make_move(self, action, player):
        """
        Generate a new state that will be reached, after making an action by players at current board.

        Args:
            action (int): the index of action to take.
            player (int): the index of the player who takes the action.
        """
        pass

    def revert_move(self):
        """
        Revert last move.
        """
        self.logger = self.logger.parent

        self.logger.child.parent = None
        self.logger.child = None

    def revert_full_game(self):
        """
        Revert full game to the beginning.
        """
        while self.logger.parent is not None:
            self.revert_move()


    @abstractmethod
    def get_valid_moves(self, cur_player=1):
        """
        Returns a list of valid moves that can be executed by a player (moves written in index format)

        Args:
            cur_player (int): the index of the current player.

        Returns:
            valid_moves (np.array(int)): list of valid moves in indexes.
        """
        pass

    @abstractmethod
    def get_next_player(self, action, player):
        """
        Returns the next player that will take the action.
        On the base of the last move.

        Args:
            action (int): the index of the last taken action.
            player (int): the index of the player who took the action.

        Returns:
            (int): index of the next player
        """
        pass

    @abstractmethod
    def get_value_and_terminated(self, player):
        """
        Returns current value of the game and terminated or not is it.

        Args:
            player (int): the index of the player who took the action.

        Returns:
            value (int): value of the game.
            terminated (bool): terminated or not.
        """
        pass
