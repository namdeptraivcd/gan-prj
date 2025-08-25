import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
    
    def forward(self, x, t, context):
        # @TODO: implement t into this function
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x