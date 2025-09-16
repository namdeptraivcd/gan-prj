from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, z):
        return self.net(z)