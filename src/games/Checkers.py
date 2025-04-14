import numpy as np

from src.games.Game import Game
from src.games.LoggerNode import LoggerNode


class Checkers(Game):
    """
    Implementation of a Checkers game with parent class GAME

    Attributes:
        row_count (int): count of rows in the game.
        column_count (int): count of columns in the game.
        figures_kinds ([int]): a list of possible figure kinds in integer format.
                               (same figures for different players should have different numbers)

        self.valid_squares ([int]) : a list of squares that the game is valid for.
        index_to_move ({int -> ...}): dictionary mapping index to move. (move can be of any type)
        move_to_index ({... -> int}): dictionary mapping move to index. (move can be of any type)
        action_size (int): number of possible actions.
        game_name (str): name of the game.
        logger (LoggerNode): logger node with extra data. (additional vars:
                                                                0: amount of moves without capturing
                                                                1: "kinged" or not last moved peace
                                                                2: which piece was removed)
    """

    def __init__(self, row_count, column_count):
        """
        Constructor.
        """
        self.row_count = row_count
        self.column_count = column_count
        self.figures_kinds = self._get_figures_kinds()

        self.valid_squares = self._get_valid_squares()

        self.index_to_move = self._get_index_to_move()
        self.move_to_index = {v: k for k, v in self.index_to_move.items()}
        self.action_size = len(self.index_to_move)

        self.game_name = "Checkers"
        self.logger = LoggerNode(self._get_initial_state(), 1, -1, np.array([0, 0, 0], dtype=object),
                                 None, None)

    def _get_figures_kinds(self):
        """
        Returns a list of possible figures kinds.

        0 - empty figure
        -1, 1 - usual checker
        -2, 2 - "kinged" checker

        Returns:
            res ([int]): List of possible figures in integer format.
        """
        return [-2, -1, 0, 1, 2]

    def _get_index_to_move(self):
        """
        Create a dictionary mapping index to move. (move can be of any type)

        Returns:
            res ({}): Dictionary mapping index to move. (move can be of any type)
        """
        res = {}

        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]
        index = 0

        for row, col in self.valid_squares:
            for dr, dc in directions:
                new_row = row + dr
                new_col = col + dc

                if 0 <= new_row < self.row_count and 0 <= new_col < self.column_count:
                    start_pos = self.valid_squares.index((row, col)) + 1
                    end_pos = self.valid_squares.index((new_row, new_col)) + 1

                    res[index] = (start_pos, end_pos)
                    index = index + 1

        return res

    def _get_valid_squares(self):
        """
        Returns a list of valid squares (dark squares) on a row_count X column_count board.

        Returns:
            valid_squares (list(int)): List of valid squares on board.
        """
        valid_squares = []
        for row in range(self.row_count):
            for col in range(self.column_count):
                # Dark squares are those where (row + col) is odd
                if (row + col) % 2 == 1:
                    valid_squares.append((row, col))
        return valid_squares

    def _get_initial_state(self):
        """
        Returns the initial state of the game's board.

        Returns:
            state (np.array): 2d array of shape (rows, columns) with initial state of figures.
        """
        state = np.zeros((self.row_count, self.column_count))

        first_valid_figures = ((self.column_count // 2) * ((self.row_count - 2) // 2))
        last_valid_figures = ((self.column_count // 2) * (self.row_count)) - first_valid_figures

        for i, (row, col) in enumerate(self.valid_squares[:first_valid_figures]):  # First 12 squares for Player 2
            state[row, col] = -1
        for i, (row, col) in enumerate(self.valid_squares[last_valid_figures:]):  # Last 12 squares for Player 1
            state[row, col] = 1
        return state

    def make_move(self, action, player):
        """
        Generate a new state that will be reached, after making an action by players at current board.

        Args:
            action (int): the index of action to take.
            player (int): the index of the player who takes the action.
        """
        new_game_log = LoggerNode()
        new_game_log.last_action = action
        add_vars = np.array([0, 0, 0], dtype=object)
        add_vars[0] = self.logger.additional_vars[0] + 1

        action = self.index_to_move[action]

        start_index, end_index = action
        start_row, start_col = self.valid_squares[start_index - 1]
        end_row, end_col = self.valid_squares[end_index - 1]

        new_state = self.logger.current_state.copy()

        # Move the piece
        new_state[start_row, start_col], new_state[end_row, end_col] = new_state[end_row, end_col], new_state[
            start_row, start_col]

        # Check if a piece needs to be "kinged"
        if (player == 1 and end_row == 0) or (player == -1 and end_row == self.row_count - 1):
            new_state[end_row, end_col] = player * 2  # Kinged piece
            add_vars[1] = 1

        # Check for capture
        if abs(start_row - end_row) == 2:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            add_vars[2] = new_state[mid_row, mid_col]
            new_state[mid_row, mid_col] = 0  # Capture the opponent's piece

        new_game_log.current_state = new_state
        new_game_log.additional_vars = add_vars

        new_game_log.current_player = self.get_next_player(new_game_log.last_action, player)

        self.logger.child = new_game_log
        new_game_log.parent = self.logger

        self.logger = self.logger.child

    def _get_capture_moves(self, player):
        """
        Getting a capture moves for a given state by a given player.

        Args:
            player (int): the index of the player who takes the action.

        Returns:
            res (list(int)): list of capture moves in index format that can be made.
        """
        res = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for i, (row, col) in enumerate(self.valid_squares):
            if self.logger.current_state[row, col] == player or self.logger.current_state[row, col] == player * 2:
                for dr, dc in directions:
                    if dr * player > 0 and self.logger.current_state[row, col] != player * 2:
                        continue

                    new_row = row + dr
                    new_col = col + dc

                    if 0 <= new_row < self.row_count and 0 <= new_col < self.column_count:
                        if (self.logger.current_state[new_row, new_col] == -player or
                                self.logger.current_state[new_row, new_col] == -player * 2):
                            capture_row = new_row + dr
                            capture_col = new_col + dc

                            if (0 <= capture_row < self.row_count and
                                    0 <= capture_col < self.column_count and
                                    self.logger.current_state[capture_row, capture_col] == 0):
                                res.append(self.move_to_index[
                                               (i + 1, self.valid_squares.index((capture_row, capture_col)) + 1)])
        return res

    def _get_normal_moves(self, player):
        """
        Getting a normal moves for a given state by a given player.

        Args:
            player (int): the index of the player who takes the action.

        Returns:
            res (list(int)): list of normal moves in index format that can be made.
        """
        res = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for i, (row, col) in enumerate(self.valid_squares):
            if self.logger.current_state[row, col] == player or self.logger.current_state[row, col] == player * 2:
                for dr, dc in directions:
                    if dr * player > 0 and self.logger.current_state[row, col] != player * 2:
                        continue

                    new_row = row + dr
                    new_col = col + dc

                    if (0 <= new_row < self.row_count and
                            0 <= new_col < self.column_count and
                            self.logger.current_state[new_row, new_col] == 0):
                        res.append(self.move_to_index[(i + 1, self.valid_squares.index((new_row, new_col)) + 1)])
        return res

    def get_valid_moves(self, cur_player=1):
        """
        Returns a list of valid moves that can be executed by a player. (moves written in index format)

        Args:
            cur_player (int): the index of the current player.

        Returns:
            valid_moves (np.array(int)): list of valid moves in indexes.
        """
        player = cur_player
        valid_moves = self._get_capture_moves(player)

        if not valid_moves:
            valid_moves = self._get_normal_moves(player)

        return valid_moves

    def _get_opponent(self, player):
        """
        Getting opponent index.

        Args:
            player (int): the index of the player.

        Returns:
            (int): the index of the opponent.
        """
        return -player

    def get_next_player(self, action, player):
        """
        Returns the next player that will take the action on the base of the last move.

        Args:
            action (int): the index of the last taken action.
            player (int): the index of the player who took the action.

        Returns:
            next_player (int): index of the next player.
        """
        cur_action = self.index_to_move[action]
        next_player = self._get_opponent(player)

        start_row, _ = self.valid_squares[cur_action[0] - 1]
        end_row, _ = self.valid_squares[cur_action[1] - 1]

        if abs(start_row - end_row) >= 2:
            valid_moves = self.get_valid_moves(player)

            for i in valid_moves:
                start, end = self.index_to_move[i]

                start_row, _ = self.valid_squares[start - 1]
                end_row, _ = self.valid_squares[end - 1]

                if start == cur_action[1] and abs(start_row - end_row) >= 2:
                    next_player = player
                    break

        return next_player

    def _check_win(self, player):
        """
        Checks that opponent can make a move.

        Args:
            player (int): the index of the player who took the last action.

        Returns:
            res (boolean): True if the opponent can make a move.
        """
        opponent = self._get_opponent(player)
        return len(self.get_valid_moves(opponent)) == 0

    def get_value_and_terminated(self, player):
        """
        Returns current value of the game and terminated or not is it.

        Args:
            player (int): the index of the player who took the action.

        Returns:
            value (int): value of the game.
            terminated (bool): terminated or not.
        """
        if self._check_win(player):
            return 1, True  # Player wins

        if self.logger.additional_vars[0] >= 50:
            return 0, True

        return 0, False
