# Directory for RL Algorithms

At this directory should be placed RL Algorithm, that will be used in training and should implement *AlgorithmsTemplate*
class.

## Variables and function that should be determined and implemented

***Variables:***

* **model**: There should be used model from *models_Structures* directory, for training.
* **optimizer**: Torch optimizers, that will be helping in training our model.
* **game**: Implementation of the game from *games* directory, that will be used for training the model.
* **args**: List of arguments, that will be used in training process (Examples: batch_size, num_epochs, and etc.)
* **algorithm_name**: String with name of the algorithm, for future search of weights for model with this algorithm.

***Methods:***

* **selfPlay**: Method, that will be playing game, with usage of our RL algorithm. 
* **train**: Process of training model on the base of played games.
* **learn**: Whole process of training model with saving of the model in the end.