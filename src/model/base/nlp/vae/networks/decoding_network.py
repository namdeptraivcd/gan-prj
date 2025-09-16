from torch import nn
import torch

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def reparameterize(self, mean, standard_deviation): #TODO: tại sao cần hàm này
        epsilon = torch.randn (mean.shape)
        latent = mean + standard_deviation * epsilon #TODO: tại sao lại thế, review paper
        return latent

    def forward(self, mean, standard_deviation):
        latent = self.reparameterize(mean, standard_deviation)
        return self.net(latent)
        #TODO return torch.sigmoid(reconstructured_input) # tại sao cần sigmoid ở đây -> review paper
    