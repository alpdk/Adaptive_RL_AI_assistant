# Directory for models' Structures

This directory contain different model structures. All of them should implement a *nn.Module* interface.

Also, as inputs, models should take:

1) **game**: object that implement interface class *Game*. This class required, because every model should start from
   amount of layers, equal to the numbers of *figures_kinds*.
2) **device**: device, where the model will be placed.

Base models should return 3 variables:

1) policy: vector of move probabilities
2) move_values: median income of moves
3) value: value of the current state

For the adaptive models there are only 1 variable to return:

1) policy: vector of move probabilities