import os
import random
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import trange
from src.rl_algorithms.AlphaZero.MCTS import MCTS

from src.rl_algorithms.AlgorithmsTemplate import AlgorithmsTemplate


class AlphaZero(AlgorithmsTemplate):
    """
    This class provide a function that implements the AlphaZero algorithm.

    Attributes:
        model (nn.Module): model that will be used for training in algorithm
        optimizer (torch.optim.Optimizer): optimizer that will be used for training in algorithm
        game (Game): game that will be used for training in algorithm
        args ({}): arguments that will be passed to the algorithm
        MCTS (MCTS): MCTS that will be used for training in algorithm
        algorithm_name (str): name of the algorithm
    """

    def __init__(self, model, optimizer, game, args):
        """
        Constructor for initializing the AlphaZero class

        Args:
            model (nn.Module): model that will be used for training in algorithm
        optimizer (torch.optim.Optimizer): optimizer that will be used for training in algorithm
        game (Game): game that will be used for training in algorithm
        args ({}): arguments that will be passed to the algorithm
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.algorithm_name = "AlphaZero"

    def selfPlay(self):
        """
        Algorithm of playing game, until will be reached a terminal state of the game

        Returns:
            memory (np.array): memory of the game
        """
        memory = []
        player = 1

        while True:
            action_probs = self.mcts.search(player)

            memory.append((self.game.logger.current_state, action_probs, player))

            # temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            # action_probs = temperature_action_probs / np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=action_probs)
            self.game.make_move(action, player)

            value, is_terminal = self.game.get_value_and_terminated(player)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value

                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))

                self.game.revert_full_game()
                return returnMemory

            player = self.game.get_next_player(action, player)

    def train(self, memory):
        """
        Training our model on a base of played games played

        Args:
            memory (np.array): memory of the games
        """
        random.shuffle(memory)

        overall_mid_loss = 0

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets.view(-1, 1))

            loss = policy_loss + value_loss

            overall_mid_loss = overall_mid_loss + loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("Loss: ", overall_mid_loss / (len(memory) // self.args['batch_size'] + 1) )

    def learn(self):
        """
        Whole process of learning model on a base of played games

        In the end model and optimizer should be saved in directories "trained_models" and "trained_optimizers"
        Files should contain the name of the game, the name of the algorithm, and the structure used, all in lowercase format.
        """
        # Define directories
        model_dir = "models_weights"
        optimizer_dir = "optimizers_weights"

        for iteration in trange(self.args['num_iterations']):
            memory = []

            self.model.eval()

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()

            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            self.save_weights(self.model.state_dict(), model_dir, "model", "pth", iteration)
            self.save_weights(self.optimizer.state_dict(), optimizer_dir, "optimizer", "pt", iteration)

        self.save_weights(self.model.state_dict(), model_dir, "model", "pth")
        self.save_weights(self.optimizer.state_dict(), optimizer_dir, "optimizer", "pt")
