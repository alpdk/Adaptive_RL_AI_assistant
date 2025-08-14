import copy
import random
import time

import torch
import numpy as np
import torch.nn.functional as F

from tqdm import trange

from src.trainers.AdaptiveTrainerTemplate import AdaptiveTrainerTemplate


class AdaptTrainer(AdaptiveTrainerTemplate):
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

    def selfPlay(self, game, base_model, rl_algorithm, adaptation_algorithm):
        """
        Algorithm of playing game, until will be reached a terminal state of the game

        Args:
            game (Game): Game that will be played
            base_model (nn.Module): Base mmodel of the game
            # adaptive_model (nn.Module): Adaptive model of the game, that will be trained
            rl_algorithm (): RL algorithm
            adaptation_algorithm (AdaptTemplate): algorithm for calculation new policy

        Returns:
            memory (np.array): memory of the game
        """
        memory = []
        player = 1

        start_state = copy.deepcopy(game.state)
        logger_pointer = game.logger

        while True:
            # start_time = time.perf_counter()
            new_policy, target_value = rl_algorithm.adaptive_probs(game, base_model, adaptation_algorithm)

            original_policy, move_values, _ = base_model(torch.tensor(game.get_encoded_state(game.state),
                                                                      device=base_model.device).unsqueeze(0))

            original_policy = original_policy.squeeze(0).detach().cpu().numpy()
            move_values = move_values.squeeze(0).detach().cpu().numpy()

            memory.append((original_policy, target_value, new_policy, player))
            game.logger.set_action_probs_and_values(new_policy, move_values)

            action = np.random.choice(game.action_size, p=new_policy)
            game.make_move(action)

            value, terminated = game.get_value_and_terminated()

            if terminated:
                returnMemory = []
                for hist_original_policy, hist_target_value, hist_new_policy, hist_player in memory:
                    # hist_outcome = value if hist_player == player else -value

                    returnMemory.append((
                        hist_original_policy,
                        hist_target_value,
                        hist_new_policy,
                        hist_player
                    ))

                game.revert_move(start_state, logger_pointer)
                # algorithm.return_to_base()

                # end_time = time.perf_counter()
                #
                # elapsed_time = (end_time - first_start_time) /
                # # print(f"Operation result: {result}")
                # print(f"SelfPlay step time: {elapsed_time:.6f} seconds")

                return returnMemory

            player = game.logger.current_player

    def train(self, adaptive_model, adaptive_optimizer, memory, loss_coef):
        """
        Method for changing model weights

        Args:
            # base_model (nn.Module): Base mmodel of the game
            adaptive_model (nn.Module): Adaptive model of the game, that will be trained
            adaptive_optimizer (optim.Optimizer): Optimizer that will be used for the adaptive model
            memory (np.array): memory of the games
            loss_coef (dict): Loss coefficient
        """
        random.shuffle(memory)

        overall_mid_loss = 0
        overall_policy_loss = 0
        # overall_moves_values_loss = 0
        # overall_value_loss = 0

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            original_policy, target_value, new_policy, player = zip(*sample)

            original_policy, target_value, new_policy, player = (np.array(original_policy),
                                                                 np.array(target_value),
                                                                 np.array(new_policy),
                                                                 np.array(player))

            original_policy = torch.tensor(original_policy, dtype=torch.float32, device=adaptive_model.device)
            target_value = torch.tensor(target_value, dtype=torch.float32, device=adaptive_model.device).unsqueeze(1)
            new_policy = torch.tensor(new_policy, dtype=torch.float32,
                                                device=adaptive_model.device)
            player = torch.tensor(player, dtype=torch.float32, device=adaptive_model.device)

            input = torch.cat((original_policy, target_value), dim=1)

            # out_policy, out_value = self.model(state)
            out_policy = adaptive_model(input)

            policy_loss = F.cross_entropy(out_policy, new_policy)
            # moves_values_loss = F.l1_loss(out_moves_values, moves_values_targets)
            # moves_values_loss = F.mse_loss(out_moves_values, moves_values_targets)
            # moves_values_loss = F.pairwise_distance(out_moves_values, moves_values_targets, p=2)
            # value_loss = F.mse_loss(out_value, value_targets.view(-1, 1))

            # loss_coef = {'policy_loss': 1.0,
            #              'moves_values_loss': 1.0,
            #              'value_loss': 0.1}

            loss = loss_coef['policy_loss'] * policy_loss

            overall_mid_loss = overall_mid_loss + loss
            overall_policy_loss = overall_policy_loss + loss_coef['policy_loss'] * policy_loss
            # overall_moves_values_loss = overall_moves_values_loss + loss_coef['moves_values_loss'] * moves_values_loss
            # overall_value_loss = overall_value_loss + loss_coef['value_loss'] * value_loss

            adaptive_optimizer.zero_grad()

            # policy_loss.backward(retain_graph=True)
            # moves_values_loss.backward(retain_graph=True)
            # value_loss.backward(retain_graph=True)
            loss.backward()

            adaptive_optimizer.step()
        print("Loss: ", overall_mid_loss / (len(memory) // self.args['batch_size'] + 1))

        print(
            f"policy_loss: {loss_coef['policy_loss'] * overall_policy_loss / (len(memory) // self.args['batch_size'] + 1)}")
        # print(
        #     f"moves_values_loss: {loss_coef['moves_values_loss'] * overall_moves_values_loss / (len(memory) // self.args['batch_size'] + 1)}")
        # print(
        #     f"value_loss: {loss_coef['value_loss'] * overall_value_loss / (len(memory) // self.args['batch_size'] + 1)}")

    def learn(self, game, base_model, adaptive_model, adaptive_optimizer, rl_algorithm, adaptation_algorithm,
              loss_coef):
        """
        Method execution whole loop of training the model

        Args:
            game (Game): Game that will be played
            base_model (nn.Module): Base mmodel of the game
            adaptive_model (nn.Module): Adaptive model of the game, that will be trained
            adaptive_optimizer (optim.Optimizer): Optimizer that will be used for the adaptive model
            rl_algorithm (): RL algorithm
            adaptation_algorithm (AdaptTemplate): algorithm for calculation new policy
            loss_coef (dict): Loss coefficient
        """
        model_dir, optimizer_dir = self.save_directories()

        game_name = game.game_name
        algorithm_name = rl_algorithm.algorithm_name
        structure_name = adaptive_model.structure_name

        for iteration in trange(self.args['num_iterations']):
            memory = []

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay(game, base_model, rl_algorithm, adaptation_algorithm)

            for epoch in trange(self.args['num_epochs']):
                self.train(adaptive_model, adaptive_optimizer, memory, loss_coef)

            self.save_weights(game_name, algorithm_name, structure_name, adaptive_model.state_dict(), model_dir, "pth",
                              iteration)
            self.save_weights(game_name, algorithm_name, structure_name, adaptive_optimizer.state_dict(), optimizer_dir,
                              "pt",
                              iteration)

        self.save_weights(game_name, algorithm_name, structure_name, adaptive_model.state_dict(), model_dir, "pth")
        self.save_weights(game_name, algorithm_name, structure_name, adaptive_optimizer.state_dict(), optimizer_dir,
                          "pt")
