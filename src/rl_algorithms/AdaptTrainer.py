import random
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import trange

from src.rl_algorithms.TrainerTemplate import TrainerTemplate

class AdaptTrainer(TrainerTemplate):
    """
    This class provide code for adaptability training for models

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

        while True:
            action_probs, action_values, _ = self.base_model(
                (torch.tensor(self.game.get_encoded_state(self.game.logger.current_state),
                                 device=self.base_model.device).unsqueeze(0))
            )

            valid_moves = self.game.get_valid_moves(player)
            valid_moves_list = np.zeros(self.game.action_size, dtype=bool)
            valid_moves_list[valid_moves] = True

            action_probs = self.game.get_normal_policy(action_probs, player)
            action_values = self.game.get_normal_values(action_values, player)

            self.game.logger.set_action_probs_and_values(action_probs, action_values)

            # temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            # action_probs = temperature_action_probs / np.sum(temperature_action_probs)
            median_opponent_income, new_action_probs = self.algorithm.probs_modification()

            # new_action_probs = F.softmax(new_action_probs)
            # median_opponent_income = torch.squeeze(median_opponent_income, dim=0).cpu().detach().numpy()

            new_action_probs = new_action_probs * valid_moves_list
            new_action_probs = new_action_probs / new_action_probs.sum()

            memory.append((action_probs, new_action_probs, median_opponent_income))

            action = np.random.choice(self.game.action_size, p=new_action_probs)
            self.game.make_move(action, player)

            value, is_terminal = self.game.get_value_and_terminated(player)

            if is_terminal:
                returnMemory = []
                for hist_action_probs, hist_new_action_probs, hist_median_opponent_income in memory:
                    returnMemory.append((
                        hist_action_probs,
                        hist_new_action_probs,
                        hist_median_opponent_income
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

            action_probs, new_action_probs, median_opponent_income = zip(*sample)
            # action_probs, median_opponent_income, value_targets = zip(*sample)

            action_probs, new_action_probs, median_opponent_income = np.array(action_probs), np.array(new_action_probs), np.array(median_opponent_income)

            action_probs = torch.tensor(action_probs, dtype=torch.float32, device=self.adapt_model.device)
            new_action_probs = torch.tensor(new_action_probs, dtype=torch.float32, device=self.adapt_model.device)
            median_opponent_income = torch.tensor(median_opponent_income, dtype=torch.float32, device=self.adapt_model.device).unsqueeze(dim=1)

            # out_policy, out_value = self.model(state)
            input = np.concatenate((median_opponent_income, action_probs), axis=1)
            input = torch.tensor(input, dtype=torch.float32, device=self.adapt_model.device)
            out_policy = self.adapt_model(input)

            policy_loss = F.cross_entropy(out_policy, new_action_probs)
            # value_loss = F.mse_loss(out_value, value_targets.view(-1, 1))

            loss = policy_loss

            overall_mid_loss = overall_mid_loss + loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("Loss: ", overall_mid_loss / (len(memory) // self.args['batch_size'] + 1))

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
            self.adapt_model.eval()

            for param in self.base_model.parameters():
                param.requires_grad = False

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.adapt_model.train()

            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            self.save_weights(self.adapt_model.state_dict(), model_dir, "model", "pth", iteration)
            self.save_weights(self.optimizer.state_dict(), optimizer_dir, "optimizer", "pt", iteration)

        self.save_weights(self.adapt_model.state_dict(), model_dir, "model", "pth")
        self.save_weights(self.optimizer.state_dict(), optimizer_dir, "optimizer", "pt")
