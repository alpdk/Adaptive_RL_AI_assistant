# Adaptive RL AI assistant

This repository will provide comfortable way of combination between RL algorithms, model structures, and games, that
will provide faster way of identification of better combinations.

Inside of the src directory can be found directories with implementations of different games, RL algorithms, and model
structures. All of them should be placed inside of directories with names: games, models_structures, and rl_algorithms.

## Training your own model

For training model you should implement game, rl_algorithm, and model structure. All of them should be placed in
corresponding directories. Guides for implementation of every part can be found inside of each directory.

After that, you should run train_model_script, that require variables:

1) Name of the game
2) RL algorithm name
2) Name of the model structure

Example:

``
python train_model.py Chess ResNet AlphaZero
``

## How to compare models

For compression of models you should run scrypt `models_compression.py`. 

This scrypt require next variables:

1) Name of the game
2) Count of game, that will be played
3) Rl algorithm name from local model
4) Model structure from local model
5) RL algorithm name from external model
6) Model structure from external model

Example:

``
python models_compression.py Checkers 10 AlphaZero ResNet AlphaZero ResNet
``
