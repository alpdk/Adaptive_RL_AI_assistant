import json

import numpy as np

from abc import abstractmethod

import torch


class Game:
    """
    This is a parent class, for games, that will be used for training model

    Attributes:
        state (np.array): current state of the game
        figures_kinds ([int]): a list of possible figure kinds in integer format
                               (same figures for different players should have different numbers)
        index_to_move ({int -> ...}): dictionary mapping index to move (move can be of any type)
        move_to_index ({... -> int}): dictionary mapping move to index (move can be of any type)
        action_size (int): number of possible actions
        game_name (str): name of the game
        logger (LoggerNode): logger node
    """
    state = None
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
            res (np.array()): array with a length of all actions
        """
        res = np.zeros(self.action_size, dtype=int)
        res[valid_moves] = 1
        return res

    def get_encoded_state(self, state):
        """
        Returns the encoded state of the game in a format of boards, where every board contain
        only 1 type of the figures.

        Args:
            state (np.array): state of the game

        Returns:
            encoded_state (np.array()): 3d array of shape (len(figures_kinds), rows, columns)
        """
        pass
        # encoded_state = np.stack(
        #     [self.state == condition for condition in self.figures_kinds]
        # ).astype(np.float32)  # 2 represents the kinged pieces
        #
        # if len(self.state.shape) == 3:
        #     encoded_state = np.swapaxes(encoded_state, 0, 1)
        #
        # return encoded_state

    @abstractmethod
    def _get_figures_kinds(self):
        """
        Returns a list of possible figure kinds in integer format
        You should create your own bijection between numbers and figures
        Empty spot also classified as a figure

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
    def _get_initial_state(self):
        """
        Returns the initial state of the game's board

        Returns:
            np.array(): 2d array of shape (rows, columns) with initial state of figures
        """
        pass

    @abstractmethod
    def make_move(self, move_index):
        """
        Method for executing move in the game

        Args:
            move_index (int): index of the move to execute
        """
        pass


    @abstractmethod
    def get_valid_moves(self):
        """
        Returns a list of valid moves that can be executed for the current player (moves written in index format)

        Returns:
            valid_moves (np.array(int)): list of valid moves in indexes
        """
        pass

    @abstractmethod
    def get_next_player(self):
        """
        Method for identification of the next player

        Returns:
            player (int): next player value
        """
        pass

    @abstractmethod
    def get_value_and_terminated(self):
        """
        Returns current value of the game and terminated or not is it

        Returns:
            value (int): value of the game
            terminated (bool): is game finished
        """
        pass

    def get_normal_policy(self, policy):
        """
        Return normalized policy of moves

        Returns:
            policy (np.array): Policy of moves from current state
            player (int): index of the player who took the action
        """
        # policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

        valid_moves = self.get_valid_moves()
        valid_moves = self.get_moves_to_np_array(valid_moves)

        policy = policy * valid_moves

        if np.all(policy == 0):
            policy = valid_moves

        policy = policy / np.sum(policy)

        return policy

    def get_normal_values(self, values, player):
        """
        Return normalized policy of moves

        Returns:
            values (np.array): values of moves from current state
            player (int): index of the player who took the action
        """
        values = values.squeeze(0).detach().cpu().numpy()

        valid_moves = self.get_valid_moves()
        valid_moves = self.get_moves_to_np_array(valid_moves)

        values = values * valid_moves

        return values

    @abstractmethod
    def revert_move(self, *args, **kwargs):
        """
        Method for reverting moves
        """
        pass
    # def collect_opponent_moves(self, history_depth):
    #     """
    #     Method for collecting opponent moves
    #
    #     Args:
    #         history_depth (int): how many opponent moves to collect
    #
    #     Returns:
    #          opponent_moves (np.ndarray): opponent moves
    #     """
    #     current_player = self.logger.current_player
    #
    #     opponent_moves = np.empty(shape=(history_depth,), dtype=object)
    #     logger_ref = self.logger
    #
    #     index = 0
    #     logger_ref = logger_ref.parent
    #
    #     while index < history_depth and logger_ref is not None:
    #         if current_player != logger_ref.current_player:
    #             opponent_moves[index] = logger_ref
    #             index += 1
    #
    #         logger_ref = logger_ref.parent
    #
    #     return opponent_moves
    #
    # def get_some_history(self, history_depth):
    #     """
    #     Method for collecting input for model
    #
    #     Args:
    #         history_depth (int): how many opponent moves to collect
    #
    #     Returns:
    #         res (np_array()): encoded states hor last history_depth steps of an opponent
    #     """
    #
    #     opponent_moves = self.collect_opponent_moves()
    #
    #     new_input = np.empty(shape=(history_depth + 1, ), dtype=object)
    #
    #     new_input[0] = self.get_encoded_state(self.logger.current_state)
    #
    #     for i in range(history_depth):
    #         if opponent_moves[i] is None:
    #             new_input[i + 1] = self.get_encoded_state(np.zeros(shape=self.logger.current_state.shape, dtype=int))
    #         else:
    #             new_input[i + 1] = self.get_encoded_state(opponent_moves[i].current_state)
    #
    #     res = np.stack(new_input)
    #     res = res.reshape(-1, self.row_count, self.column_count)
    #     return res

    # @abstractmethod
    # def return_to_base(self):
    #     """
    #     This method will be returning some parameters to their base values
    #     """
    #     pass

    # def revert_move(self):
    #     """
    #     Revert last move.
    #     """
    #     self.logger = self.logger.parent
    #
    #     self.return_to_base()
    #
    #     self.logger.child.parent = None
    #     self.logger.child = None

    # def revert_full_game(self):
    #     """
    #     Revert full game to the beginning
    #     """
    #     while self.logger.parent is not None:
    #         self.revert_move()

    # @abstractmethod
    # def create_current_key(self):
    #     """
    #     Transforms data, that will be used as a key, to some format
    #
    #     Returns:
    #          res (string): new format, for the json
    #     """
    #     pass
    #
    # @abstractmethod
    # def create_current_value(self, probs, value, move_values):
    #     """
    #     Transforms data, that will be used as a value, to some format
    #
    #     Parameters:
    #         probs (np.array): values of moves from current state
    #         value (integer): value of the current state
    #         move_values (np.array): values of moves from current state
    #
    #     Returns:
    #          res (): new format, for the json
    #     """
    #     pass
    #
    # @abstractmethod
    # def transform_value_to_data(self, value):
    #     """
    #     Transforms value from the json to our actual data
    #
    #     Parameters:
    #         value ([]): some value, that should be parsed
    #
    #     Returns:
    #         probs (np.array): values of moves from current state
    #         value (int): value of the current state
    #         move_values (np.array): values of moves from current state
    #     """
    #     pass

    # def create_current_key(self):
    #     """
    #     Transforms data, that will be used as a key, to some format
    #
    #     Returns:
    #          res (string): new format, for the json
    #     """
    #     state = self.logger.current_state.tolist()
    #     player = self.logger.current_player
    #
    #     data = [state, player]
    #     return json.dumps(data)
    #
    # def create_current_value(self, probs, move_values):
    #     """
    #     Transforms data, that will be used as a value, to some format
    #
    #     Parameters:
    #         probs (np.array): values of moves from current state
    #         move_values (np.array): values of moves from current state
    #
    #     Returns:
    #          res (): new format, for the json
    #     """
    #     return [probs.tolist(), move_values.tolist()]
    #
    # def transform_value_to_data(self, value):
    #     """
    #     Transforms value from the json to our actual data
    #
    #     Parameters:
    #         value ([]): some value, that should be parsed
    #
    #     Returns:
    #         res ([]):
    #             probs (np.array): values of moves from current state
    #             move_values (np.array): values of moves from current state
    #     """
    #     probs = np.array(value[0])
    #     move_values = np.array(value[1])
    #
    #     return [probs, move_values]
