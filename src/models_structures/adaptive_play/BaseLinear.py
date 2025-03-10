import torch.nn as nn
import torch.nn.functional as F

class BaseLinear(nn.Module):
    """
    Class that provide a model, that will be trained for adaptation to the opponent level

    Attributes:
        device (): Device that the model will be trained on
        startBlock (nn.Sequential): first block of the model
        backBone (nn.Sequential): backbone of the model constructed from LinearBlocks
        finalBlock (nn.Sequential): final block of the model
        structure_name (string): name of the model structure
    """
    def __init__(self, game, num_linearBlock, num_hidden, device):
        """
        Initializer

        Args:
            game (Game): Game instance
            num_linearBlock (int): Number of linear blocks
            num_hidden (int): Number of neurons in hidden layer
            device (torch.device): Device that the model will be trained on
        """
        super().__init__()

        self.device = device

        self.startBlock = nn.Sequential(
            nn.Linear((game.action_size + 1), num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [LinearBlock(num_hidden) for i in range(num_linearBlock)]
        )

        self.finalBlock = nn.Sequential(
            nn.Linear((game.action_size + 1), num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.Softmax(dim=1)
        )

        self.to(device)

        self.structure_name = "BaseLinear"

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            policy (np.array[]): adapted policy of moves from regular probs and opponent moves coefficient
        """
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        policy = self.finalBlock(x)
        return policy

class LinearBlock(nn.Module):
    """
    LinearBlock, that will be used in model

    Attributes:
        layer (nn.Sequential): linear layer with batch normalization and relu activation
    """
    def __init__(self, num_hidden):
        """
        Initializer

        Args:
            num_hidden (int): Number of neurons in hidden layer
        """
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            x: Result of a linear block
        """
        residual = x
        x = self.layer(x)
        x += residual
        x = F.relu(x)
        return x





