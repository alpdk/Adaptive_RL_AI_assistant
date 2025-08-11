import copy
import math
import random
import time

import numpy as np
import torch


class TreeNode:
    """
    Class representing a node in a tree

    Attributes:
        args ({}): Some arguments, that will be passed used
        player (int): Current player
        action_taken (int): Index of the action taken
        parent (TreeNode): Parent node
        children ([TreeNode]): List of children
        visit_count (int): Amount of visits of the node
        value_sum (float): Sum of the node values
        policy ([float]): Probabilities to select a child as a next move
    """

    def __init__(self, args=None, player=1, action_taken=None, parent=None, children=None):
        """
        Constructor

        Args:
            action_taken (int): Index of the action taken
            player (int): Current player
            parent (TreeNode): Parent node
            children ([TreeNode]): List of children
        """
        if children is None:
            children = []

        self.args = args
        self.player = player
        self.action_taken = action_taken
        self.parent = parent
        self.children = children

        self.visit_count = 0
        self.value_sum = 0

        self.policy = None

    def is_fully_expanded(self):
        """
        Check if the node is fully expanded (all children are defined)

        Returns:
             (boolean): True if the node is fully expanded, False otherwise
        """
        return len(self.children) > 0

    def get_gpuct(self, child, move_index):
        """
        Calculate the ucb value of a child node.

        Args:
            child (Node): The child node
            move_index (int): Move index for reaching child node

        Returns:
            ucb (int): The ucb value of a child node.
        """
        if child.visit_count == 0:
            return np.inf

        q_value = child.value_sum / (child.visit_count + 1)

        return q_value + self.policy[move_index] * self.args['C'] * math.exp(
            self.args['tau'] * math.log(self.visit_count)) / (child.visit_count + 1)

        # q_value = child.value_sum / (child.visit_count + 1)
        #
        # return q_value + self.policy[move_index] * self.args['C'] * math.exp(self.args['tau'] * math.log(self.visit_count)) / (child.visit_count + 1)

    def select(self):
        """
        Algorithm that return the best node with highest gpuct value

        Returns:
            best_child (Node): The best node with highest gpuct value
        """
        best_children = []
        best_gpuct = -np.inf

        for child in self.children:
            gpuct = self.get_gpuct(child, child.action_taken)

            if gpuct > best_gpuct:
                best_children = [child]
                best_gpuct = gpuct
            elif gpuct == best_gpuct:
                best_children.append(child)

        random_best_child = random.choice(best_children)

        return random_best_child

    def expand(self, policy, game):
        """
        Method that create children by a policy, related to player

        Args:
            policy (np.array[]): A numpy array representing the policy of making move by a player
            game (Game): game that is used
        """
        game_state = copy.deepcopy(game.state)
        logger_pointer = game.logger

        for action, prob in enumerate(policy):
            if prob > 0:
                game.make_move(action)
                next_player = game.logger.current_player
                game.revert_move(game_state, logger_pointer)

                child = TreeNode(self.args, next_player, action, self)
                self.children.append(child)

    def backpropagation(self, value, whose_value, current_root):
        """
        Method that backpropagation a value to all parents.

        Parameters:
            value (float): value of the move
            whose_value (int): player whose value we back propagate
            current_root (TreeNode): current tree node of the tree
        """
        self.visit_count += 1

        if self != current_root:
            if self.parent.player == whose_value:
                self.value_sum += value
            else:
                self.value_sum -= value

            self.parent.backpropagation(value, whose_value, current_root)


class MCTS:
    """
    Class to implement Monte Carlo Tree Search (MCTS)

    Attributes:
        args ({}): Some arguments, that will be passed used
        root (TreeNode): Root node of the MCTS tree
        current_node (TreeNode): Current node that will be used
        algorithm_name (str): Name of the algorithm
    """

    def __init__(self, args):
        """
        Constructor

        Args:
            args ({}): Some arguments, that will be passed used
            algorithm_name (str): Name of the algorithm
        """
        self.args = args
        self.root = TreeNode(args)
        self.current_node = self.root
        self.algorithm_name = "MCTS"

    def calc_probs(self, game, values=None):
        """
        Method for calculating the probabilities of each action

        Parameters:
            game (Game): Game that is used
            values (np.array[float]): array of values

        Returns:
            np.array[]: probabilities of each action
        """
        action_probs = np.zeros(game.action_size)

        original_actions = action_probs

        for child in self.current_node.children:
            action = child.action_taken
            # action_probs[action] = math.exp(values[action])
            action_probs[action] = math.pow(max(values[action] + 1, 0.05), 2) / max((1 - values[action]), 0.05)

        if np.sum(action_probs) == 0:
            print("What the hell!!!")
            print(original_actions)

        action_probs /= np.sum(action_probs)

        return action_probs

    def calc_values(self, game):
        """
        Method for calculating the values of each action

        Parameters:
             game (Game): Game that is used

        Returns:
            np.array[]: values of each action
        """
        action_values = np.zeros(game.action_size)
        for child in self.current_node.children:
            action = child.action_taken
            action_values[action] = child.value_sum / child.visit_count

        return action_values

    def search(self, game, model):
        """
        Method that performs MCTS search.

        Args:
            game (Game): Game that is used
            model (nn.Module): Model that will be used

        Returns:
            action_probs (np.array): action probabilities
            action_values (np.array): action values
        """
        # start_time = time.perf_counter()
        policy, _, value = model(torch.tensor(game.get_encoded_state(game.state),
                                                   device=model.device).unsqueeze(0))

        policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

        policy = (1 - self.args['dirichlet_epsilon']) * policy + np.random.dirichlet(
            [self.args['dirichlet_alpha']] * game.action_size) * self.args['dirichlet_epsilon']

        valid_moves = game.get_valid_moves()
        valid_moves = game.get_moves_to_np_array(valid_moves)

        policy = policy * valid_moves
        policy = game.get_normal_policy(policy)

        # policy = policy / np.sum(policy)

        if not self.current_node.is_fully_expanded():
            self.current_node.expand(policy, game)

        while self.current_node.visit_count < self.args['num_searches']:
            node = self.current_node

            save_state = copy.deepcopy(game.state)
            logger_pointer = game.logger

            while node.is_fully_expanded():
                if node.policy is None:
                    policy, _, _ = model(torch.tensor(game.get_encoded_state(game.state),
                                                           device=model.device).unsqueeze(0))
                    policy = policy.squeeze(0).detach().cpu().numpy()

                    valid_moves = game.get_valid_moves()
                    valid_moves = game.get_moves_to_np_array(valid_moves)

                    policy = policy * valid_moves
                    policy = game.get_normal_policy(policy)

                    node.policy = policy

                node = node.select()
                game.make_move(node.action_taken)

            last_move_player = node.parent.player
            value, terminated = game.get_value_and_terminated()

            if not terminated:
                policy, _, value = model(torch.tensor(game.get_encoded_state(game.state),
                                                       device=model.device).unsqueeze(0))
                cur_player = node.player

                policy = policy.squeeze(0).detach().cpu().numpy()
                valid_moves = game.get_valid_moves()
                valid_moves = game.get_moves_to_np_array(valid_moves)

                policy = policy * valid_moves
                policy = game.get_normal_policy(policy)
                value = value.item()

                node.expand(policy, game)

            # if value == None:

            node.backpropagation(value, last_move_player, self.current_node)
            game.revert_move(save_state, logger_pointer)

        action_values = self.calc_values(game)
        # valid_moves = game.get_valid_moves()
        # valid_moves = game.get_moves_to_np_array(valid_moves)
        #
        # action_values = action_values * valid_moves
        action_probs = self.calc_probs(game, action_values)
        # end_time = time.perf_counter()

        # elapsed_time = end_time - start_time
        # # print(f"Operation result: {result}")
        # print(f"Search time: {elapsed_time:.6f} seconds")

        return action_probs, action_values

    def change_current_node(self, action_taken):
        """
        Method that changes the current node

        Args:
            action_taken (index): Move index tha twas executed
        """
        for child in self.current_node.children:
            if child.action_taken == action_taken:
                self.current_node = child
                break

    def return_to_root(self):
        """
        Method for returning tree to the root
        """
        self.current_node = self.root
