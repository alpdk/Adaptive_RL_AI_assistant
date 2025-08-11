import copy

import torch
import numpy as np

from abc import abstractmethod

from sqlalchemy import column
from sympy.physics.units import length

from src.games.Game import Game
from src.games.LoggerNode import LoggerNode


class UltimateTicTacToe(Game):
    """
    Implementation of a Ultimate TicTacToe game with parent class GAME

    Attributes:
        row_count (int): count of rows in the game.
        column_count (int): count of columns in the game.
        figures_kinds ([int]): a list of possible figure kinds in integer format.
                               (same figures for different players should have different numbers)

        index_to_move ({int -> ...}): dictionary mapping index to move. (move can be of any type)
        move_to_index ({... -> int}): dictionary mapping move to index. (move can be of any type)
        action_size (int): number of possible actions.
        game_name (str): name of the game.
        logger (LoggerNode): logger node with extra data. (additional vars:
                                                                0: what field forced to use from the parent node
                                                                1: which subfield was used by a player
                                                                2: win states on a big field)
    """

    def __init__(self, row_column_count=3):
        """
        Constructor

        Args:
            row_column_count (int): count of rows and columns in board and smaller boards
        """
        self.row_count = row_column_count
        self.column_count = row_column_count

        self.state = self._get_initial_state()
        self.figures_kinds = self._get_figures_kinds()

        self.index_to_move = self._get_index_to_move()
        self.move_to_index = {v: k for k, v in self.index_to_move.items()}
        self.action_size = len(self.index_to_move)

        self.game_name = "UltimateTicTacToe"
        self.logger = LoggerNode()

    def get_row(self):
        """
        Method for returning the row_count value

        Returns:
            row_count (int): count of rows in the game
        """
        return self.row_count * self.row_count

    def get_column(self):
        """
        Method for returning the column_count value

        Returns:
            column_count (int): count of rows in the game
        """
        return self.column_count * self.column_count + 2

    def _get_figures_kinds(self):
        """
        Returns a list of possible figure kinds in integer format.

        -1 - circle
        0 - empty figure
        1 - cross

        Returns:
            [int]: List of possible figures in integer format
        """
        return [-1, 0, 1]

    def _get_index_to_move(self):
        """
        Create a dictionary mapping index to move. (move can be of any type)

        Returns:
            res ({}): Dictionary mapping index to move. (move can be of any type)
        """
        res = {}

        index = 0

        for field_row in range(self.row_count):
            for field_column in range(self.column_count):
                for subfield_row in range(self.row_count):
                    for subfield_column in range(self.column_count):
                        board = field_row * self.column_count + field_column
                        cell = subfield_row * self.column_count + subfield_column

                        res[index] = (board, cell)
                        index = index + 1

        return res

    def _get_initial_state(self):
        """
        Returns the initial state of the game's board

        Returns:
            np.array(): 2d array of shape (rows * rows, columns * columns) with initial state of figures
        """
        state = np.zeros((self.row_count * self.row_count + 2, self.column_count * self.column_count))
        state[-2] = -1
        state[-1] = 0
        return state

    def _check_horizontal_line(self, board, cell, player):
        """
        Method for checking horizontal line winning condition

        Args:
            board (int): index of the board for check
            cell (int): index of the cell for check
            player (int): player who made a move

        Returns:
            res (bool): is there a winning condition or not
        """
        start_cell = cell // self.column_count * self.column_count

        for i in range(self.column_count):
            if self.state[board][start_cell + i] != player:
                return False

        return True

    def _check_vertical_line(self, board, cell, player):
        """
        Method for checking vertical line winning condition

        Args:
            board (int): index of the board for check
            cell (int): index of the cell for check
            player (int): player who made a move

        Returns:
            res (bool): is there a winning condition or not
        """
        start_cell = cell % self.column_count

        for i in range(self.row_count):
            if self.state[board][start_cell + i * self.column_count] != player:
                return False

        return True

    def _check_left_diagonal_line(self, board, cell, player):
        """
        Method for checking left diagonal line winning condition

        Args:
            board (int): index of the board for check
            cell (int): index of the cell for check
            player (int): player who made a move

        Returns:
            res (bool): is there a winning condition or not
        """
        start_cell = cell % self.column_count - cell // self.column_count

        if start_cell != 0:
            return False

        for i in range(self.row_count):
            if self.state[board][i * (self.column_count + 1)] != player:
                return False

        return True

    def _check_right_diagonal_line(self, board, cell, player):
        """
        Method for checking right diagonal line winning condition

        Args:
            board (int): index of the board for check
            cell (int): index of the cell for check
            player (int): player who made a move

        Returns:
            res (bool): is there a winning condition or not
        """
        start_cell = cell % self.column_count + cell // self.column_count

        if start_cell != self.column_count - 1:
            return False

        for i in range(self.row_count):
            if self.state[board][start_cell + i * (self.column_count - 1)] != player:
                return False

        return True

    def _check_board_win(self, board, cell, player):
        """
        Method for checking winning condition for the last changed board

        Args:
            board (int): index of the board for check
            cell (int): index of the cell for check
            player (int): player who made a move

        Returns:
            res (bool): is there a winning condition or not
        """
        hor_res = self._check_horizontal_line(board, cell, player)
        ver_res = self._check_vertical_line(board, cell, player)
        left_res = self._check_left_diagonal_line(board, cell, player)
        right_res = self._check_right_diagonal_line(board, cell, player)

        return hor_res or ver_res or left_res or right_res

    def _check_board_usability(self, board):
        """
        Method for checking is there any free spots to move on the board

        Args:
            board (int): index of the board for check

        Returns:
            res (bool): is there a free spots or not
        """
        for i in range(self.column_count * self.row_count):
            if self.state[board][i] == 0:
                return True

        return False

    def _get_small_board_valid_moves(self, board):
        """
        Method for collecting valid moves from the small board

        Args:
            board (int): index of the board for check

        Returns:
            res (list): list of valid moves
        """
        res = []

        for i in range(self.column_count * self.row_count):
            if self.state[board][i] == 0:
                res.append(board * self.column_count * self.row_count + i)

        return res

    def get_valid_moves(self):
        """
        Returns a list of valid moves that can be executed for the current player (moves written in index format)

        Returns:
            valid_moves (np.array(int)): list of valid moves in indexes
        """
        res = []

        if int(self.state[-2][0]) == -1:
            for i in range(self.column_count * self.row_count):
                if self.state[-1][i] == 0:
                    res = res + self._get_small_board_valid_moves(i)

            return res

        return self._get_small_board_valid_moves(int(self.state[-2][0]))

    def get_next_player(self):
        """
        Method for identification of the next player

        Returns:
            player (int): next player value
        """
        return -self.logger.current_player

    def make_move(self, move_index):
        """
        Method for executing move in the game

        Args:
            move_index (int): index of the move to execute
        """
        board, cell = self.index_to_move[move_index]

        if move_index not in self.get_valid_moves():
            print(f"Move {move_index} is not valid!")
            return

        changes = {}
        changes['move_index'] = move_index

        self.state[board][cell] = self.logger.current_player

        # check winning small board
        if self._check_board_win(board, cell, self.logger.current_player):
            self.state[-1][board] = self.logger.current_player
        else:
            self.state[-1][board] = 0 if self._check_board_usability(board) else None

        # identify next small board to move
        if self.state[-1][cell] == 0:
            self.state[-2] = cell
        else:
            self.state[-2] = -1

        changes['board_res'] = self.state[-1][board]
        changes['next_board'] = self.state[-2]
        # changes['prev_moving_player'] = self.logger.current_player

        new_logger_node = LoggerNode(self.get_next_player(), changes, self.logger)
        self.logger.child = new_logger_node
        self.logger = new_logger_node

    def get_value_and_terminated(self):
        """
        Returns current value of the game and terminated or not is it

        Returns:
            value (int): value of the game
            terminated (bool): is game finished
        """
        board = self.logger.changes["move_index"] // (self.column_count * self.row_count)

        if self._check_board_win(-1, board, self.logger.parent.current_player):
            return (self.logger.parent.current_player, True)

        return (None, False) if self._check_board_usability(-1) else (0, True)

    def get_encoded_state(self, state):
        """
        Returns the encoded state of the game in a format of boards, where every board contain
        only 1 type of the figures.

        Args:
            state (np.array): state of the game

        Returns:
            encoded_state (np.array()): 3d array of shape (len(figures_kinds), rows, columns)
        """
        pointer = state[9, 0]

        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        for i in range(len(encoded_state)):
            encoded_state[i, 9] = pointer

        return encoded_state

    def revert_move(self, state, logger_pointer):
        """
        Method for returning to specific state of the game

        Args:
            state (np.array): State of the game for returning
            logger_pointer (LoggerNode): Pointer to the logger node
        """
        self.state = copy.deepcopy(state)

        self.logger = logger_pointer
        self.logger.child.parent = None
        self.logger.child = None

    # def _get_index_to_move(self):
    #     """
    #     Create a dictionary mapping index to move. (move can be of any type)
    #
    #     Returns:
    #         res ({}): Dictionary mapping index to move. (move can be of any type)
    #     """
    #     res = {}
    #
    #     index = 0
    #
    #     for field_row in range(self.row_count):
    #         for field_column in range(self.column_count):
    #             for subfield_row in range(self.row_count):
    #                 for subfield_column in range(self.column_count):
    #                     row = field_row * self.row_count + subfield_row
    #                     column = field_column * self.column_count + subfield_column
    #
    #                     res[index] = (row, column)
    #                     index = index + 1
    #
    #     return res
    #
    # def _get_initial_state(self):
    #     """
    #     Returns the initial state of the game's board
    #
    #     Returns:
    #         np.array(): 2d array of shape (rows * rows, columns * columns) with initial state of figures
    #     """
    #     state = np.zeros((self.row_count * self.row_count + 2, self.column_count * self.column_count))
    #     state[9] = -1
    #     state[10] = 0
    #     return state
    #
    # def _check_line(self, field, row_start, column_start, row_change, column_change, player):
    #     """
    #     This method is how long is the line starting from changed cell
    #
    #     Parameters:
    #         field ([]): game field
    #         row_start (int): row where was made a move
    #         column_start (int): column where was made a move
    #         row_change (int): what direction will be used for changing row
    #         column_change (int): what direction will be used for changing column
    #         player (int): player who made a move
    #     """
    #     line = 1
    #
    #     row, column = row_start + row_change, column_start + column_change
    #
    #     while 0 <= row < self.column_count and 0 <= column < self.row_count and field[row, column] == player:
    #         line += 1
    #         row += row_change
    #         column += column_change
    #
    #     row, column = row_start - row_change, column_start - column_change
    #
    #     while 0 <= row < self.column_count and 0 <= column < self.row_count and field[row, column] == player:
    #         line += 1
    #         row -= row_change
    #         column -= column_change
    #
    #     return line
    #
    # def _check_field_win(self, field, move, player):
    #     """
    #     This method check field condition
    #
    #     Parameters:
    #         field ([]): field for checking
    #         move (int): where is the change
    #         player (int): who made a move
    #
    #     Returns:
    #          res (int): field result
    #     """
    #     row, column = self.index_to_move[move]
    #
    #     max_in_line = max(max(self._check_line(field, row, column, 1, 0, player),
    #                           self._check_line(field, row, column, 0, 1, player)),
    #                       max(self._check_line(field, row, column, 1, 1, player),
    #                           self._check_line(field, row, column, 1, -1, player)))
    #
    #     if max_in_line == min(self.row_count, self.column_count):
    #         return player
    #
    #     for i in range(self.row_count):
    #         for j in range(self.column_count):
    #             if field[i, j] != player and field[i][j] != -player:
    #                 return None
    #
    #     return 0
    #
    # def make_move(self, action, player):
    #     """
    #     Generate a new state that will be reached, after making an action by players at current board
    #
    #     Args:
    #         action (int): the index of action to take
    #         player (int): the index of the player who takes the action
    #     """
    #     new_game_log = LoggerNode()
    #     new_game_log.last_action = action
    #     add_vars = [None, None, None]
    #
    #     row_move, column_move = self.index_to_move[action]
    #
    #     # Creating new state
    #     new_state = self.logger.current_state.copy()
    #     new_state[row_move, column_move] = player
    #     new_game_log.current_state = new_state
    #
    #     # What cell was used from the big field
    #     self.logger.additional_vars[1] = action // (self.row_count * self.column_count)
    #
    #     # What cell will be used from the big field
    #     first_subfield_action = self.logger.additional_vars[1] * (self.row_count * self.column_count)
    #     allowed_field = action - first_subfield_action
    #     add_vars[0] = allowed_field
    #
    #     # Create field for checking condition
    #     test_subfield = np.zeros((self.row_count, self.column_count), dtype=int)
    #
    #     for i in range(first_subfield_action, first_subfield_action + self.row_count * self.column_count):
    #         row, column = self.index_to_move[i]
    #
    #         test_subfield[row % self.row_count, column % self.column_count] = new_state[row, column]
    #
    #     # Checking subfield result and save them
    #     new_field_res = self.logger.additional_vars[2].copy()
    #     field_row, field_column = self.logger.additional_vars[1] // self.column_count, self.logger.additional_vars[1] % self.column_count
    #     new_field_res[field_row, field_column] = self._check_field_win(test_subfield,  action % (self.row_count * self.column_count), player)
    #     add_vars[2] = new_field_res
    #
    #
    #     if add_vars[2][add_vars[0] // self.column_count, add_vars[0] % self.column_count] != None:
    #         add_vars[0] = -1
    #
    #     # Save new_game_log node
    #     new_game_log.additional_vars = add_vars
    #     new_game_log.current_player = self.get_next_player(new_game_log.last_action, player)
    #
    #     self.logger.child = new_game_log
    #     new_game_log.parent = self.logger
    #
    #     self.logger = self.logger.child
    #
    # # def revert_full_game(self):
    # #     """
    # #     Revert full game to the beginning
    # #     """
    # #     while self.logger.parent is not None:
    # #         self.revert_move()
    # #
    # #     self.logger.additional_vars[0] = -1
    # #     self.logger.additional_vars[1] = np.full((self.row_count, self.column_count), None, dtype=object)
    #
    # # def return_to_base(self):
    # #     """
    # #     This method will be returning some parameters to their base values
    # #     """
    # #     self.logger.additional_vars[1] = None
    #
    # def get_valid_moves(self, cur_player=1):
    #     """
    #     Returns a list of valid moves that can be executed by a player (moves written in index format)
    #
    #     Args:
    #         cur_player (int): the index of the current player
    #
    #     Returns:
    #         valid_moves (np.array(int)): list of valid moves in indexes
    #     """
    #     valid_moves = []
    #
    #     field_cell = self.logger.additional_vars[0]
    #
    #     if field_cell == -1 or self.logger.additional_vars[2][field_cell // self.column_count, field_cell % self.column_count] is not None:
    #         i = 0
    #
    #         while i < self.row_count * self.column_count * self.row_count * self.column_count:
    #             row, column = self.index_to_move[i]
    #             field_row, field_column = row // self.row_count, column // self.column_count
    #
    #             if self.logger.additional_vars[2][field_row, field_column] != None:
    #                 i += self.row_count * self.column_count
    #                 continue
    #
    #             if self.logger.current_state[row, column] == 0:
    #                 valid_moves.append(self.move_to_index[(row, column)])
    #
    #             i = i + 1
    #     else:
    #         field_row, field_column = self.index_to_move[field_cell]
    #
    #         move_index = field_row * (self.column_count * self.row_count) * self.column_count + \
    #                       field_column * (self.column_count * self.row_count)
    #
    #         for row in range(self.row_count):
    #             for column in range(self.column_count):
    #                 move = self.index_to_move[move_index]
    #
    #                 if self.logger.current_state[move[0], move[1]] == 0:
    #                     valid_moves.append(move_index)
    #
    #                 move_index += 1
    #
    #     return valid_moves
    #
    # def get_next_player(self, action, player):
    #     """
    #     Returns the next player that will take the action
    #     On the base of the last move
    #
    #     Args:
    #         action (int): the index of the last taken action
    #         player (int): the index of the player who took the action
    #
    #     Returns:
    #         (int): index of the next player
    #     """
    #     return -player
    #
    # def get_value_and_terminated(self, player):
    #     """
    #     Returns current value of the game and terminated or not is it
    #
    #     Args:
    #         player (int): the index of the player who took the action
    #
    #     Returns:
    #         value (int): value of the game
    #         terminated (bool): terminated or not
    #     """
    #     state = self.logger.additional_vars[2]
    #     move = self.logger.parent.additional_vars[1]
    #
    #     row, column = self.index_to_move[move]
    #
    #     if state[row, column] == None:
    #         return None, False
    #
    #     res = self._check_field_win(state, move, player)
    #
    #     if player == res:
    #         return 1, True
    #
    #     for i in range(self.row_count * self.column_count):
    #         row, column = self.index_to_move[i]
    #
    #         if state[row, column] == None:
    #             return None, False
    #
    #     cur_player, opponent_player = 0, 0
    #
    #     for i in range(self.row_count):
    #         for j in range(self.column_count):
    #             if state[i, j] == player:
    #                 cur_player += 1
    #             elif state[i, j] == -player:
    #                 opponent_player += 1
    #
    #     if cur_player > opponent_player:
    #         res = 1
    #     elif cur_player < opponent_player:
    #         res = -1
    #     else:
    #         res = 0
    #
    #     return res, True
