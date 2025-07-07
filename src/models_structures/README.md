# Directory for models' Structures

This directory contain different model structures. All of them should implement a *nn.Module* interface.

Also, as inputs, models should take:

1) **game**: object that implement interface class *Game*. This class required, because every model should start from
   amount of layers, equal to the numbers of *figures_kinds*.
2) **device**: device, where the model will be placed.