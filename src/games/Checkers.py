import numpy as np

from .Game import Game

class Checkers(Game):
    """
    Implementation of a Checkers game with parent class GAME

    Attributes:
        row_count (int): count of rows in the game
        column_count (int): count of columns in the game
        figures_kinds ([int]): a list of possible figure kinds in integer format
                               (same figures for different players should have different numbers)

        self.valid_squares ([int]) : a list of squares that the game is valid for
        index_to_move ({int -> ...}): dictionary mapping index to move (move can be of any type)
        move_to_index ({... -> int}): dictionary mapping move to index (move can be of any type)
        action_size (int): number of possible actions
    """

    def __init__(self):
        """
        Constructor
        """
        self.row_count = 8
        self.column_count = 8
        self.figures_kinds = self._get_figures_kinds()

        self.valid_squares = self._get_valid_squares()

        self.index_to_move = self._get_index_to_move()
        self.move_to_index = {v: k for k, v in self.index_to_move.items()}
        self.action_size = len(self.index_to_move)

    def _get_figures_kinds(self):
        """
        Returns a list of possible figures kinds.

        0 - empty figure
        -1, 1 - usual checker
        -2, 2 - "kinged" checker

        Returns:
            [int]: List of possible figures in integer format
        """
        return [-2, -1, 0, 1, 2]

    def _get_index_to_move(self):
        """
        Create a dictionary mapping index to move (move can be of any type)

        Returns:
            {}: Dictionary mapping index to move (move can be of any type)
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
        Returns a list of valid squares (dark squares) on an row_count X column_count board

        Returns:
            list(int): List of valid squares on board
        """
        valid_squares = []
        for row in range(self.row_count):
            for col in range(self.column_count):
                # Dark squares are those where (row + col) is odd
                if (row + col) % 2 == 1:
                    valid_squares.append((row, col))
        return valid_squares

    def get_initial_state(self):
        """
        Returns the initial state of the game's board

        Returns:
            np.array(): 2d array of shape (rows, columns) with initial state of figures
        """
        state = np.zeros((self.row_count, self.column_count))

        first_valid_figures = ((self.column_count // 2) * ((self.row_count - 2) // 2))
        last_valid_figures = ((self.column_count // 2) * (self.row_count)) - first_valid_figures

        for i, (row, col) in enumerate(self.valid_squares[:first_valid_figures]):  # First 12 squares for Player 2
            state[row, col] = -1
        for i, (row, col) in enumerate(self.valid_squares[last_valid_figures:]):  # Last 12 squares for Player 1
            state[row, col] = 1
        return state

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
        action = self.index_to_move[action]

        start_index, end_index = action
        start_row, start_col = self.valid_squares[start_index - 1]
        end_row, end_col = self.valid_squares[end_index - 1]

        new_state = state.copy()

        # Move the piece
        new_state[start_row, start_col], new_state[end_row, end_col] = new_state[end_row, end_col], new_state[start_row, start_col]

        # Check if a piece needs to be "kinged"
        if (player == 1 and end_row == 0) or (player == -1 and end_row == self.row_count - 1):
            new_state[end_row, end_col] = player * 2  # Kinged piece

        # Check for capture
        if abs(start_row - end_row) == 2:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            new_state[mid_row, mid_col] = 0  # Capture the opponent's piece

        return new_state

    def get_capture_moves(self, state, player):
        """
        Getting a capture moves for a given state by a given player

        Args:
            state (np.array): 2d array of shape (rows, columns)
            player (int): the index of the player who takes the action

        Returns:
            list(int): list of capture moves in index format that can be made
        """
        res = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for i, (row, col) in enumerate(self.valid_squares):
            if state[row, col] == player or state[row, col] == player * 2:
                for dr, dc in directions:
                    if dr * player > 0 and state[row, col] != player * 2:
                        continue

                    new_row = row + dr
                    new_col = col + dc

                    if 0 <= new_row < self.row_count and 0 <= new_col < self.column_count:
                        if state[new_row, new_col] == -player or state[new_row, new_col] == -player * 2:
                            capture_row = new_row + dr
                            capture_col = new_col + dc

                            if 0 <= capture_row < self.row_count and 0 <= capture_col < self.column_count and state[capture_row, capture_col] == 0:
                                res.append(self.move_to_index[(i + 1, self.valid_squares.index((capture_row, capture_col)) + 1)])
        return res

    def get_normal_moves(self, state, player):
        """
        Getting a normal moves for a given state by a given player

        Args:
            state (np.array): 2d array of shape (rows, columns)
            player (int): the index of the player who takes the action

        Returns:
            list(int): list of normal moves in index format that can be made
        """
        res = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for i, (row, col) in enumerate(self.valid_squares):
            if state[row, col] == player or state[row, col] == player * 2:
                for dr, dc in directions:
                    if dr * player > 0 and state[row, col] != player * 2:
                        continue

                    new_row = row + dr
                    new_col = col + dc

                    if 0 <= new_row < self.row_count and 0 <= new_col < self.column_count and state[new_row, new_col] == 0:
                        res.append(self.move_to_index[(i + 1, self.valid_squares.index((new_row, new_col)) + 1)])
        return res

    def get_valid_moves(self, state, cur_player=1):
        """
        Returns a list of valid moves that can be executed by a player (moves written in index format)

        Args:
            state (np.array): 2d array of shape (rows, columns)
            cur_player (int): the index of the current player

        Returns:
            valid_moves (np.array(int)): list of valid moves in indexes
        """
        player = cur_player
        valid_moves = self.get_capture_moves(state, player)

        if valid_moves == []:
            valid_moves = self.get_normal_moves(state, player)

        return valid_moves

    def get_opponent(self, player):
        """
        Getting opponent index.

        Args:
            player (int): the index of the player

        Returns:
            (int): the index of the opponent
        """
        return -player

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
        cur_action = self.index_to_move[action]
        next_player = self.get_opponent(player)

        eat_step_size = (self.column_count // 2) * 2 - 1

        if abs(cur_action[0] - cur_action[1]) >= eat_step_size:
            valid_moves = self.get_valid_moves(state, player)

            for i in valid_moves:
                start, end = self.index_to_move[i]

                if start == cur_action[1] and abs(start - end) >= eat_step_size:
                    next_player = player
                    break

        return next_player

    def check_win(self, state, player):
        """
        Checks that opponent can make a move

        Args:
            state (np.array): current game state
            player (int): the index of the player who took the last action

        Returns:
            boolean: True if the opponent can make a move
        """
        opponent = self.get_opponent(player)
        return len(self.get_valid_moves(state, opponent)) == 0

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
        if self.check_win(state, player):
            return 1, True  # Player wins
        return 0, False