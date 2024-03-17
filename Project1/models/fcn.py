import torch
from torch import nn


class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32 * 32 * 3),
            nn.Tanh(),
        )

    def forward(self, x, train=True):
        x1 = self.encoder1(x)

        # add gaussian noise during training
        if train:
            x1 = x1 + torch.randn(x1.size()).cuda() * 0.05

        out = self.decoder(x1)
        return out
