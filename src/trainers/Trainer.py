import copy
import random
import time

import torch
import numpy as np
import torch.nn.functional as F

from tqdm import trange

from src.trainers.BaseTrainerTemplate import BaseTrainerTemplate


class Trainer(BaseTrainerTemplate):
    """
    This class provide whole training loop for the RL models

    Attributes:
        args ({}): Dictionary of arguments, that will be used for training
    """

    def __init__(self, args):
        """
        Constructor

        Args:
            args ({}): Dictionary of arguments, that will be used for training
        """
        super().__init__(args)

    def selfPlay(self, game, model, algorithm):
        """
        Algorithm of playing game, until will be reached a terminal state of the game

        Args:
            game (Game): Game that will be played
            model (nn.Module): Model that will be trained
            algorithm (): RL algorithm

        Returns:
            memory (np.array): memory of the game
        """
        memory = []
        player = 1

        start_state = copy.deepcopy(game.state)
        logger_pointer = game.logger

        first_start_time = time.perf_counter()

        while True:
            # start_time = time.perf_counter()
            action_probs, moves_values = algorithm.search(game, model)

            memory.append((game.state, action_probs, moves_values, game.logger.current_player))
            game.logger.set_action_probs_and_values(action_probs, moves_values)

            # temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            # action_probs = temperature_action_probs / np.sum(temperature_action_probs)
            action = np.random.choice(game.action_size, p=action_probs)
            game.make_move(action)
            # layer += 1

            algorithm.algorithm_step(action)

            value, terminated = game.get_value_and_terminated()

            if terminated:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_moves_values, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value

                    returnMemory.append((
                        game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_moves_values,
                        hist_player,
                        hist_outcome
                    ))

                game.revert_move(start_state, logger_pointer)
                algorithm.return_to_base()

                # end_time = time.perf_counter()
                #
                # elapsed_time = (end_time - first_start_time) /
                # # print(f"Operation result: {result}")
                # print(f"SelfPlay step time: {elapsed_time:.6f} seconds")

                return returnMemory

            player = game.logger.current_player

            # end_time = time.perf_counter()
            #
            # elapsed_time = end_time - start_time
            # # print(f"Operation result: {result}")
            # print(f"SelfPlay step time: {elapsed_time:.6f} seconds")

    def train(self, model, optimizer, memory, loss_coef):
        """
        Method for changing model weights

        Args:
            model (nn.Module): Model that will be trained
            optimizer (optim.Optimizer): Optimizer that will be used
            memory (np.array): memory of the games
            loss_coef (dict): Loss coefficient
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

            state = torch.tensor(state, dtype=torch.float32, device=model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=model.device)
            moves_values_targets = torch.tensor(moves_values_targets, dtype=torch.float32,
                                                device=model.device)
            player = torch.tensor(player, dtype=torch.float32, device=model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=model.device)

            # out_policy, out_value = self.model(state)
            out_policy, out_moves_values, out_value = model(state)

            """
            Make module for loss functions
            """

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            # moves_values_loss = F.l1_loss(out_moves_values, moves_values_targets)
            moves_values_loss = F.mse_loss(out_moves_values, moves_values_targets)
            # moves_values_loss = F.pairwise_distance(out_moves_values, moves_values_targets, p=2)
            value_loss = F.mse_loss(out_value, value_targets.view(-1, 1))

            # loss_coef = {'policy_loss': 1.0,
            #              'moves_values_loss': 1.0,
            #              'value_loss': 0.1}

            loss = (loss_coef['policy_loss'] * policy_loss +
                    loss_coef['moves_values_loss'] * moves_values_loss +
                    loss_coef['value_loss'] * value_loss)

            overall_mid_loss = overall_mid_loss + loss
            overall_policy_loss = overall_policy_loss + loss_coef['policy_loss'] * policy_loss
            overall_moves_values_loss = overall_moves_values_loss + loss_coef['moves_values_loss'] * moves_values_loss
            overall_value_loss = overall_value_loss + loss_coef['value_loss'] * value_loss

            optimizer.zero_grad()

            # policy_loss.backward(retain_graph=True)
            # moves_values_loss.backward(retain_graph=True)
            # value_loss.backward(retain_graph=True)
            loss.backward()

            optimizer.step()
        print("Loss: ", overall_mid_loss / (len(memory) // self.args['batch_size'] + 1))

        print(f"policy_loss: {loss_coef['policy_loss'] * overall_policy_loss / (len(memory) // self.args['batch_size'] + 1)}")
        print(f"moves_values_loss: {loss_coef['moves_values_loss'] * overall_moves_values_loss / (len(memory) // self.args['batch_size'] + 1)}")
        print(f"value_loss: {loss_coef['value_loss'] * overall_value_loss / (len(memory) // self.args['batch_size'] + 1)}")

    def learn(self, game, model, optimizer, algorithm, loss_coef):
        """
        Method execution whole loop of training the model

        Args:
            game (Game): Game that will be played
            model (nn.Module): Model that will be trained
            optimizer (optim.Optimizer): Optimizer that will be used
            algorithm (): RL algorithm
            loss_coef (dict): Loss coefficient
        """
        model_dir, optimizer_dir = self.save_directories()

        game_name = game.game_name
        algorithm_name = algorithm.algorithm_name
        structure_name = model.structure_name

        for iteration in trange(self.args['num_iterations']):
            memory = []

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay(game, model, algorithm)

            # self.clear_save_states()

            for epoch in trange(self.args['num_epochs']):
                self.train(model, optimizer, memory, loss_coef)

            self.save_weights(game_name, algorithm_name, structure_name, model.state_dict(), model_dir, "pth", iteration)
            self.save_weights(game_name, algorithm_name, structure_name, optimizer.state_dict(), optimizer_dir, "pt", iteration)

        self.save_weights(game_name, algorithm_name, structure_name, model.state_dict(), model_dir, "pth")
        self.save_weights(game_name, algorithm_name, structure_name, optimizer.state_dict(), optimizer_dir, "pt")