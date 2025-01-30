import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    """
    Class that provide a model, that will be trained.

    Attributes:
        device (): Device that the model will be trained on.
        startBlock (nn.Sequential): first block of the model.
        backBone (nn.Sequential): backbone of the model constructed from ResBlocks.
        policyHead (nn.Sequential): policy head of the model.
        valueHead (nn.Sequential): value head of the model.
    """
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(len(game.figures_kinds), num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, len(game.figures_kinds), kernel_size=3, padding=1),
            nn.BatchNorm2d(len(game.figures_kinds)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(len(game.figures_kinds) * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        """
        Forward pass of the model.

        Returns:
            policy (np.array[]): policy of moves from current state.
            value (int): value from current state.
        :param x:
        :return:
        """
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    """
    ResBlock, that will be used in model

    Attributes:
        conv1 (nn.Conv2d): first convolution layer.
        bn1 (nn.BatchNorm2d): first batch normalization layer.
        conv2 (nn.Conv2d): second convolution layer.
        bn2 (nn.BatchNorm2d): second batch normalization layer.
    """
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Returns:
            x: Result of a residual block.
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x