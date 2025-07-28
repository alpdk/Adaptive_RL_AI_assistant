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

    def __init__(self, row_count, column_count):
        """
        Constructor

        Parameters:
            row_count (int): count of rows in field and subfields.
            column_count (int): count of columns in field and subfields.
        """
        self.row_count = row_count
        self.column_count = column_count
        self.figures_kinds = self._get_figures_kinds()

        self.index_to_move = self._get_index_to_move()
        self.move_to_index = {v: k for k, v in self.index_to_move.items()}
        self.action_size = len(self.index_to_move)

        self.game_name = "UltimateTicTacToe"
        field_res = np.full((self.row_count, self.column_count), None, dtype=object)
        self.logger = LoggerNode(self._get_initial_state(), 1, -1, [-1, None, field_res], None, None)

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
                        row = field_row * self.row_count + subfield_row
                        column = field_column * self.column_count + subfield_column

                        res[index] = (row, column)
                        index = index + 1

        return res

    def _get_initial_state(self):
        """
        Returns the initial state of the game's board

        Returns:
            np.array(): 2d array of shape (rows * rows, columns * columns) with initial state of figures
        """
        state = np.zeros((self.row_count * self.row_count + 2, self.column_count * self.column_count))
        state[9] = -1
        state[10] = 0
        return state

    def _check_line(self, field, row_start, column_start, row_change, column_change, player):
        """
        This method is how long is the line starting from changed cell

        Parameters:
            field ([]): game field
            row_start (int): row where was made a move
            column_start (int): column where was made a move
            row_change (int): what direction will be used for changing row
            column_change (int): what direction will be used for changing column
            player (int): player who made a move
        """
        line = 1

        row, column = row_start + row_change, column_start + column_change

        while 0 <= row < self.column_count and 0 <= column < self.row_count and field[row, column] == player:
            line += 1
            row += row_change
            column += column_change

        row, column = row_start - row_change, column_start - column_change

        while 0 <= row < self.column_count and 0 <= column < self.row_count and field[row, column] == player:
            line += 1
            row -= row_change
            column -= column_change

        return line

    def _check_field_win(self, field, move, player):
        """
        This method check field condition

        Parameters:
            field ([]): field for checking
            move (int): where is the change
            player (int): who made a move

        Returns:
             res (int): field result
        """
        row, column = self.index_to_move[move]

        max_in_line = max(max(self._check_line(field, row, column, 1, 0, player),
                              self._check_line(field, row, column, 0, 1, player)),
                          max(self._check_line(field, row, column, 1, 1, player),
                              self._check_line(field, row, column, 1, -1, player)))

        if max_in_line == min(self.row_count, self.column_count):
            return player

        for i in range(self.row_count):
            for j in range(self.column_count):
                if field[i, j] != player and field[i][j] != -player:
                    return None

        return 0

    def make_move(self, action, player):
        """
        Generate a new state that will be reached, after making an action by players at current board

        Args:
            action (int): the index of action to take
            player (int): the index of the player who takes the action
        """
        new_game_log = LoggerNode()
        new_game_log.last_action = action
        add_vars = [None, None, None]

        row_move, column_move = self.index_to_move[action]

        # Creating new state
        new_state = self.logger.current_state.copy()
        new_state[row_move, column_move] = player
        new_game_log.current_state = new_state

        # What cell was used from the big field
        self.logger.additional_vars[1] = action // (self.row_count * self.column_count)

        # What cell will be used from the big field
        first_subfield_action = self.logger.additional_vars[1] * (self.row_count * self.column_count)
        allowed_field = action - first_subfield_action
        add_vars[0] = allowed_field

        # Create field for checking condition
        test_subfield = np.zeros((self.row_count, self.column_count), dtype=int)

        for i in range(first_subfield_action, first_subfield_action + self.row_count * self.column_count):
            row, column = self.index_to_move[i]

            test_subfield[row % self.row_count, column % self.column_count] = new_state[row, column]

        # Checking subfield result and save them
        new_field_res = self.logger.additional_vars[2].copy()
        field_row, field_column = self.logger.additional_vars[1] // self.column_count, self.logger.additional_vars[1] % self.column_count
        new_field_res[field_row, field_column] = self._check_field_win(test_subfield,  action % (self.row_count * self.column_count), player)
        add_vars[2] = new_field_res


        if add_vars[2][add_vars[0] // self.column_count, add_vars[0] % self.column_count] != None:
            add_vars[0] = -1

        # Save new_game_log node
        new_game_log.additional_vars = add_vars
        new_game_log.current_player = self.get_next_player(new_game_log.last_action, player)

        self.logger.child = new_game_log
        new_game_log.parent = self.logger

        self.logger = self.logger.child

    # def revert_full_game(self):
    #     """
    #     Revert full game to the beginning
    #     """
    #     while self.logger.parent is not None:
    #         self.revert_move()
    #
    #     self.logger.additional_vars[0] = -1
    #     self.logger.additional_vars[1] = np.full((self.row_count, self.column_count), None, dtype=object)

    def return_to_base(self):
        """
        This method will be returning some parameters to their base values
        """
        self.logger.additional_vars[1] = None

    def get_valid_moves(self, cur_player=1):
        """
        Returns a list of valid moves that can be executed by a player (moves written in index format)

        Args:
            cur_player (int): the index of the current player

        Returns:
            valid_moves (np.array(int)): list of valid moves in indexes
        """
        valid_moves = []

        field_cell = self.logger.additional_vars[0]

        if field_cell == -1 or self.logger.additional_vars[2][field_cell // self.column_count, field_cell % self.column_count] is not None:
            i = 0

            while i < self.row_count * self.column_count * self.row_count * self.column_count:
                row, column = self.index_to_move[i]
                field_row, field_column = row // self.row_count, column // self.column_count

                if self.logger.additional_vars[2][field_row, field_column] != None:
                    i += self.row_count * self.column_count
                    continue

                if self.logger.current_state[row, column] == 0:
                    valid_moves.append(self.move_to_index[(row, column)])

                i = i + 1
        else:
            field_row, field_column = self.index_to_move[field_cell]

            move_index = field_row * (self.column_count * self.row_count) * self.column_count + \
                          field_column * (self.column_count * self.row_count)

            for row in range(self.row_count):
                for column in range(self.column_count):
                    move = self.index_to_move[move_index]

                    if self.logger.current_state[move[0], move[1]] == 0:
                        valid_moves.append(move_index)

                    move_index += 1

        return valid_moves

    def get_next_player(self, action, player):
        """
        Returns the next player that will take the action
        On the base of the last move

        Args:
            action (int): the index of the last taken action
            player (int): the index of the player who took the action

        Returns:
            (int): index of the next player
        """
        return -player

    def get_value_and_terminated(self, player):
        """
        Returns current value of the game and terminated or not is it

        Args:
            player (int): the index of the player who took the action

        Returns:
            value (int): value of the game
            terminated (bool): terminated or not
        """
        state = self.logger.additional_vars[2]
        move = self.logger.parent.additional_vars[1]

        row, column = self.index_to_move[move]

        if state[row, column] == None:
            return None, False

        res = self._check_field_win(state, move, player)

        if player == res:
            return 1, True

        for i in range(self.row_count * self.column_count):
            row, column = self.index_to_move[i]

            if state[row, column] == None:
                return None, False

        cur_player, opponent_player = 0, 0

        for i in range(self.row_count):
            for j in range(self.column_count):
                if state[i, j] == player:
                    cur_player += 1
                elif state[i, j] == -player:
                    opponent_player += 1

        if cur_player > opponent_player:
            res = 1
        elif cur_player < opponent_player:
            res = -1
        else:
            res = 0

        return res, True

