import random
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import trange

from src.rl_algorithms.base_rl_algorithms.MCTS.MCTS import MCTS
from src.rl_algorithms.TrainerTemplate import TrainerTemplate


class BaseTrainer(TrainerTemplate):
    """
    This class provide a function that implements the AlphaZero algorithm.

    Attributes:
        base_model (nn.Module): model that will be used for training in algorithm
        adapt_model (nn.Module): model that will adapt to opponent's level
        optimizer (torch.optim.Optimizer): optimizer that will be used for training in algorithm
        game (Game): game that will be used for training in algorithm
        algorithm (): algorithm, that will be used in training
        args ({}): arguments that will be passed to the algorithm
    """

    def selfPlay(self):
        """
        Algorithm of playing game, until will be reached a terminal state of the game

        Returns:
            memory (np.array): memory of the game
        """
        memory = []
        player = 1
        layer = 0

        while True:
            data = self.check_existing_layers(layer)

            if data is None:
                action_probs, moves_values = self.algorithm.search(player)
                self.save_data_in_layer(action_probs, moves_values, layer)
            else:
                action_probs, moves_values = data[0], data[1]

            memory.append((self.game.logger.current_state, action_probs, moves_values, player))

            # temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            # action_probs = temperature_action_probs / np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=action_probs)
            self.game.make_move(action, player)
            layer += 1

            value, is_terminal = self.game.get_value_and_terminated(player)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_moves_values, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value

                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_moves_values,
                        hist_player,
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
        overall_policy_loss = 0
        overall_moves_values_loss = 0
        overall_value_loss = 0

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, moves_values_targets, player, value_targets = zip(*sample)

            state, policy_targets, moves_values_targets, player, value_targets = (np.array(state),
                                                                                  np.array(policy_targets),
                                                                                  np.array(moves_values_targets),
                                                                                  np.array(player),
                                                                                  np.array(value_targets))

            state = torch.tensor(state, dtype=torch.float32, device=self.base_model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.base_model.device)
            moves_values_targets = torch.tensor(moves_values_targets, dtype=torch.float32,
                                                device=self.base_model.device)
            player = torch.tensor(player, dtype=torch.float32, device=self.base_model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.base_model.device)

            # out_policy, out_value = self.model(state)
            out_policy, out_moves_values, out_value = self.base_model(state)

            """
            Make module for loss functions
            """

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            # moves_values_loss = F.l1_loss(out_moves_values, moves_values_targets)
            moves_values_loss = F.mse_loss(out_moves_values, moves_values_targets)
            # moves_values_loss = F.pairwise_distance(out_moves_values, moves_values_targets, p=2)
            value_loss = F.mse_loss(out_value, value_targets.view(-1, 1))

            loss_coef = {'policy_loss': 1.0,
                          'moves_values_loss': 1.0,
                          'value_loss': 0.1}

            loss = (loss_coef['policy_loss'] * policy_loss +
                    loss_coef['moves_values_loss'] * moves_values_loss +
                    loss_coef['value_loss'] * value_loss)

            overall_mid_loss = overall_mid_loss + loss
            overall_policy_loss = overall_policy_loss + loss_coef['policy_loss'] * policy_loss
            overall_moves_values_loss = overall_moves_values_loss + loss_coef['moves_values_loss'] * moves_values_loss
            overall_value_loss = overall_value_loss + loss_coef['value_loss'] * value_loss

            self.optimizer.zero_grad()

            # policy_loss.backward(retain_graph=True)
            # moves_values_loss.backward(retain_graph=True)
            # value_loss.backward(retain_graph=True)
            loss.backward()

            self.optimizer.step()
        print("Loss: ", overall_mid_loss / (len(memory) // self.args['batch_size'] + 1))

        print(f"policy_loss: {overall_policy_loss / (len(memory) // self.args['batch_size'] + 1)}")
        print(f"moves_values_loss: {overall_moves_values_loss / (len(memory) // self.args['batch_size'] + 1)}")
        print(f"value_loss: {overall_value_loss / (len(memory) // self.args['batch_size'] + 1)}")

    def learn(self):
        """
        Whole process of learning model on a base of played games

        In the end model and optimizer should be saved in directories "trained_models" and "trained_optimizers"
        Files should contain the name of the game, the name of the algorithm, and the structure used, all in lowercase format.
        """
        model_dir, optimizer_dir = self.save_directories()

        for iteration in trange(self.args['num_iterations']):
            memory = []

            self.base_model.eval()

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.clear_save_states()

            self.base_model.train()

            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            self.save_weights(self.base_model.state_dict(), model_dir, "model", "pth", iteration)
            self.save_weights(self.optimizer.state_dict(), optimizer_dir, "optimizer", "pt", iteration)

        self.save_weights(self.base_model.state_dict(), model_dir, "model", "pth")
        self.save_weights(self.optimizer.state_dict(), optimizer_dir, "optimizer", "pt")
