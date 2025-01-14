import numpy as np

from .Game import Game

class Checkers(Game):
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.figures_kinds = self._get_figures_kinds()

        self.valid_squares = self._get_valid_squares()

        self.index_to_move = self._get_index_to_move()
        self.move_to_index = {v: k for k, v in self.index_to_move.items()}
        self.action_size = len(self.index_to_move)

    def _get_figures_kinds(self):
        return [-2, -1, 0, 1, 2]

    def _get_index_to_move(self):
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
        """Returns a list of valid squares (dark squares) on an 8x8 board"""
        valid_squares = []
        for row in range(self.row_count):
            for col in range(self.column_count):
                # Dark squares are those where (row + col) is odd
                if (row + col) % 2 == 1:
                    valid_squares.append((row, col))
        return valid_squares

    def get_initial_state(self):
        # Initial state: set up pieces for each player
        # Player 1 is represented by 1, and Player 2 by -1
        state = np.zeros((self.row_count, self.column_count))
        for i, (row, col) in enumerate(self.valid_squares[:12]):  # First 12 squares for Player 2
            state[row, col] = -1
        for i, (row, col) in enumerate(self.valid_squares[20:]):  # Last 12 squares for Player 1
            state[row, col] = 1
        return state

    def get_next_state(self, state, action, player):
        # Action: a tuple (start_index, end_index) in the range 1-32
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
        player = cur_player
        valid_moves = self.get_capture_moves(state, player)

        if valid_moves == []:
            valid_moves = self.get_normal_moves(state, player)

        return valid_moves

    def get_next_player(self, state, action, player):
        cur_action = self.index_to_move[action]
        next_player = self.get_opponent(player)

        if abs(cur_action[0] - cur_action[1]) >= 7:
            valid_moves = self.get_valid_moves(state, player)

            for i in valid_moves:
                start, end = self.index_to_move[i]

                if start == cur_action[1] and abs(start - end) >= 7:
                    next_player = player
                    break

        return next_player

    def check_win(self, state, player):
        # A player wins if the opponent has no valid moves left
        opponent = self.get_opponent(player)
        return len(self.get_valid_moves(state, opponent)) == 0

    def get_value_and_terminated(self, state, player):
        if self.check_win(state, player):
            return 1, True  # Player wins
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            [state == condition for condition in self.figures_kinds]
        ).astype(np.float32)  # 2 represents the kinged pieces

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state