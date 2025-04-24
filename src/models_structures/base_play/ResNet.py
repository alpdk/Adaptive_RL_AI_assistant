import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    """
    Class that provide a model, that will be trained for move probability prediction.

    Attributes:
        device (): Device that the model will be trained on
        startBlock (nn.Sequential): first block of the model
        backBone1 (nn.Sequential): backbone of the model constructed from ResBlocks
        policyHead (nn.Sequential): policy head of the model
        movesValueHead (nn.Sequential): moves value head of the model
        valueHead (nn.Sequential): value head of the model
        structure_name (string): name of the model structure
    """

    def __init__(self, game, num_resBlocks, num_hidden, device):
        """
        Initializer

        Args:
            game (Game): Game that the model will be trained on
            num_resBlocks (int): Number of ResBlocks
            num_hidden (int): Number of neurons in hidden layer
            device (torch.device): Device that the model will be trained on
        """
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(len(game.figures_kinds), num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone1 = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.get_column() * game.get_row(), game.action_size)
        )

        self.movesValueHead = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.get_column() * game.get_row(), game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, len(game.figures_kinds), kernel_size=3, padding=1),
            nn.BatchNorm2d(len(game.figures_kinds)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(len(game.figures_kinds) * game.get_column() * game.get_row(), 1),
            nn.Tanh()
        )

        self.to(device)

        self.structure_name = "ResNet"

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            policy (np.array[]): policy of moves from current state.
            move_values (np.array[]): moves value from current state.
            value (int): value from current state.
        """
        x = self.startBlock(x)
        for resBlock in self.backBone1:
            x = resBlock(x)
        policy = self.policyHead(x)
        move_values = self.movesValueHead(x)
        value = self.valueHead(x)
        return policy, move_values, value


class ResBlock(nn.Module):
    """
    ResBlock, that will be used in model

    Attributes:
        layer1 (nn.Sequential): first convolution layer with batch normalization.
        layer2 (nn.Sequential): second convolution layer with batch normalization.
    """

    def __init__(self, num_hidden):
        """
        Initializer

        Args:
            num_hidden (int): Number of neurons in hidden layer
        """
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden)
        )

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            x: Result of a residual block.
        """
        residual = x
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = x + residual
        x = F.relu(x)
        return x
