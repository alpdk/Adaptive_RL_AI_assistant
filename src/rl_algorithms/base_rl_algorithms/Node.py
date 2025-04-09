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
         policy (np.array[]): Policy for moves from this node.
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

        self.policy = None

    def is_fully_expanded(self):
        """
        Check if the node is fully expanded (all children are defined)

        Returns:
             (boolean): True if the node is fully expanded, False otherwise.
        """
        return len(self.children) > 0

    def select(self):
        """
        Algorithm that return the best node with highest gpuct value.

        Returns:
            best_child (Node): The best node with highest gpuct value.
        """
        best_child = None
        best_gpuct = -np.inf

        for child in self.children:
            gpuct = self.get_gpuct(child, child.action_taken)
            if gpuct > best_gpuct:
                best_child = child
                best_gpuct = gpuct

        return best_child

    def get_gpuct(self, child, move_index):
        """
        Calculate the ucb value of a child node.

        Parameters:
            child (Node): The child node.
            move_index (int): Move index for reaching child node.

        Returns:
            ucb (int): The ucb value of a child node.
        """
        if child.visit_count == 0:
            return np.inf

        q_value = child.value_sum

        return q_value + self.policy[move_index] * self.args['C'] * math.exp(self.args['tau'] * math.log(self.visit_count)) / (child.visit_count + 1)

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

    def backpropagation(self, value, whose_value):
        """
        Method that backpropagation a value to all parents.

        Parameters:
            value (float): value of the move
            whose_value (int): player whose value we back propagate
        """
        self.visit_count += 1

        if self.parent is not None:
            if self.parent.player == whose_value:
                self.value_sum += value
            else:
                self.value_sum -= value

            self.game.revert_move()
            self.parent.backpropagation(value, whose_value)
