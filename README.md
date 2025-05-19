# Adaptive RL AI assistant

This repository will provide comfortable way of combination between RL algorithms, model structures, and games, that
will provide faster way of identification of better combinations.

Inside of the src directory can be found directories with implementations of different games, RL algorithms, and model
structures. All of them should be placed inside of directories with names: games, models_structures, and rl_algorithms.

## Training your own model

For training model you should implement game, rl_algorithm, and model structure. All of them should be placed in
corresponding directories. Guides for implementation of every part can be found inside of each directory.

After that, you should run train_model, that require variables:

1) Name of the game
2) Name of the base model structure
3) RL algorithm name
4) Load existing weights or not (0 - no, 1 - yes)


Example:

``
python train_base_model.py Checkers ResNet MCTS 0
``

For the adaptive training you need different parameters:

1) Name of the game
2) Name of the base model structure
3) Name of the adaptive model structure
4) RL algorithm name
5) Adaptive algorithm
6) How many opponent moves use for metric calculation (positive integer)
7) Opponents move scale of relevance (float between 0.0 and 1.0)

``
python train_adaptive_model.py Checkers ResNet BaseLinear MCTS AdaptProbs 5 1.0
``

## How to compare models

For compression of models you should run scrypt `models_comparison.py`. 

This scrypt require next variables for comparison of base models:

1) Name of the game
2) Count of game, that will be played
3) Rl algorithm name for the first model
4) Model structure for the first model
5) Rl algorithm name for the second model
6) Base model comparison or adaptive (0 - base, 1 - adaptive)

Example:

``
python models_comparison.py Checkers 1000 MCTS ResNet MCTS ResNet 0
``

Moreover, the file for the first model should be placed in the `base_models_weights` directory, 
while the weights for the second model should be stored in the `external_base_weights` directory.

For comparison the adaptive model files, corresponding files should also be located in the `adapt_models_weights` and 
`external_adapt_weights` directories.

Additionally, you will need the following parameters, while the sixth parameter equal to 1:

1) Name of the adaptive algorithm
2) Name of the adaptive algorithm used for training first model
3) Model structure, that was used for the training of the first adaptive model
4) Name of the adaptive algorithm used for training second model
5) Model structure, that was used for the training of the second adaptive model

``
python models_comparison.py Checkers 1000 MCTS ResNet MCTS ResNet 1 --adaptive_algorithm AdaptProbs --local_adapt_algorithm_name AdaptProbs --local_adapt_model_structure BaseLinear --external_adapt_algorithm_name AdaptProbs --external_adapt_model_structure BaseLinear
``

As a result of the compression, you will see information about the number of games that ended in a win, loss, or draw from the perspective of Player 1.
