# Directory for Games

This directory contains implementation of games, that can be used for training of agents.
All of the game should implement Game interface and stored here.

## What kind of games can be used for training an agent.

This interface provide abilities to implement games, that have predefined field and countable moves that can be taken by
players.

Examples of board games, that can be implemented with such interface:

1) Checkers
2) Ultimate Tic-Tac-Toe

## Variables and function that should be determined and implemented

***Variables:***

* **row_count and column_count**: integers that identify size of the field.
* **figures_kinds**: list of indexes, where every index represent unique figure from the board. Same figures for
  different players should be identified by different indexes.
* **index_to_move**: dictionary that make correlation from unique indexes to unique moves. All possible moves should be
  represented at this dictionary.
* **move_to_index**: reverse version of *index_to_move* dictionary.
* **action_size**: amount of possible moves of every figure from every position on the field. It should be equal to the
  length of *index_to_move* and *move_to_index* dictionaries
* **game_name**: string that contain name of the game. This string will be used in saving model weights and for search
  of game class.
* **logger**: logger of the game, that will help to steps back in game history.

***Methods:***

* **get_row**: method for getting amount of rows in game.
* **get_column**: method for getting amount of columns in game.
* **_get_figures_kinds**: method that return list of every possible figures in game.
* **_get_index_to_move**: method that return generate *index_to_move* dictionary.
* **_get_initial_state**: method for getting clear board of the game.
* **make_move**: method that implement move execution.
* **get_valid_moves**: method that return np.array of integers, where possible moves are marked by 1, otherwise by 0.
  As input it receive current state of the game and current player.
* **get_next_player**: method that return index of next player. As input it takes current state of the game, player, and
  action that should be executed.
* **get_value_and_terminated**: method that return winner and is game in terminate state. As input ir require current
  state and last moved player.




