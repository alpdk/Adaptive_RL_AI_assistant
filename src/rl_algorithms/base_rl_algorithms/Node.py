import math
import numpy as np


class Node:
    """
    This class implement a node class, that will be used in MCTS algorithm.

    Attributes:
         game (Game): The game that will be played.
         args ({}): Some arguments, that will be passed used.
         player (int): Player number, who's turn is now.
         parent (Node): The parent of this node.
         action_taken (int): Action taken by previous player.
         children ([Node]): Children of this node.
         visit_count (int): Visit count for this node.
         value_sum (int): Value sum for this node.
    """

    def __init__(self, game, args, player=0, parent=None, action_taken=None, visit_count=0):
        self.game = game
        self.args = args
        self.player = player
        self.parent = parent
        self.action_taken = action_taken

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        """
        Check if the node is fully expanded (all children are defined)

        Returns:
             (boolean): True if the node is fully expanded, False otherwise.
        """
        return len(self.children) > 0

    def select(self, player):
        """
        Algorithm that return the best node with highest ucb value.

        Parameters:
            player(int): current player

        Returns:
             (Node): The best node with highest ucb value.
        """
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child, player)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child, player):
        """
        Calculate the ucb value of a child node.

        Parameters:
            child (Node): The child node.
            player(int): current player

        Returns:
            ucb (int): The ucb value of a child node.
        """
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = child.value_sum / (child.visit_count + 1)

        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / (child.visit_count + 1))

    def expand(self, policy, player):
        """
        Method that create children by a policy, related to player

        Parameters:
            policy (np.array[]): A numpy array representing the policy of making move by a player.
            player (int): A player, that will be making move.
        """
        for action, prob in enumerate(policy):
            if prob > 0:
                self.game.make_move(action, player)
                next_player = self.game.logger.current_player
                self.game.revert_move()

                child = Node(self.game, self.args, next_player, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value, whos_value):
        """
        Method that back propagate a value to all parents.

        Parameters:
            value (float): value of the move
            whos_value (int): player who's value we back propagate
        """
        self.visit_count += 1

        if self.parent is not None:
            if self.parent.player == whos_value:
                self.value_sum += value
            else:
                self.value_sum -= value

            self.game.revert_move()
            self.parent.backpropagate(value, whos_value)
