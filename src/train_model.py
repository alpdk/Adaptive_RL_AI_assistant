import torch

from games.Checkers import Checkers
from src.models_structures.ResNet import ResNet
from src.models.AlphaZero import AlphaZero

checkers = Checkers(8, 8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = ResNet(checkers, 4, 64, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 8,
    'num_selfPlay_iterations': 50,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alphaZero = AlphaZero(model, optimizer, checkers, args)
alphaZero.learn()