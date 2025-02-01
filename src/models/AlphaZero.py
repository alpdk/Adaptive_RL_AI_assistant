import os
import random
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import trange
from src.rl_algorithms.MCTS import MCTS

from .ModelTemplates import ModelTemplates

class AlphaZero(ModelTemplates):
    """
    This class provide a function that implements the AlphaZero algorithm.

    Attributes:
        model (nn.Module): model that will be used for training in algorithm
        optimizer (torch.optim.Optimizer): optimizer that will be used for training in algorithm
        game (Game): game that will be used for training in algorithm
        args ({}): arguments that will be passed to the algorithm
        MCTS (MCTS): MCTS that will be used for training in algorithm
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
        self.model_name = "AlphaZero"

    def selfPlay(self):
        """
        Algorithm of playing game, until will be reached a terminal state of the game

        Returns:
            memory (np.array): memory of the game
        """
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = state.copy()
            action_probs = self.mcts.search(neutral_state, player)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            # action_probs = temperature_action_probs / np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, player)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value

                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_next_player(state, action, player)

    def train(self, memory):
        """
        Training our model on a base of played games played

        Args:
            memory (np.array): memory of the games
        """
        random.shuffle(memory)

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        """
        Whole process of learning model on a base of played games

        In the end model and optimizer should be saved in directories "trained_models" and "trained_optimizers"
        Files should contain name of the model, and used structure.
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

            # Ensure the directories exist
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(optimizer_dir, exist_ok=True)

            # Save model and optimizer states
            torch.save(self.model.state_dict(), os.path.join(model_dir, f"model_{self.model_name}_{self.model.structure_name}_{iteration}.pth"))
            torch.save(self.optimizer.state_dict(), os.path.join(optimizer_dir, f"optimizer_{self.model_name}_{self.model.structure_name}.pt"))

        torch.save(self.model.state_dict(), os.path.join(model_dir, f"model_{self.model_name}_{self.model.structure_name}.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(optimizer_dir, f"optimizer_{self.model_name}_{self.model.structure_name}.pt"))