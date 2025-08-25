import torch.nn as nn
from src.model.gan.networks.generator import Generator
from src.model.gan.networks.discriminator import Discriminator
from src.config.config import GloalConfig, GANConfig


class GAN:
    def __init__(self, real_embeddings, noise_dim=GANConfig.noise_dim):
        self.device = GloalConfig.DEIVCE
        self.embed_dim = real_embeddings.size(1)
        self.real_embeddings = real_embeddings.to(self.device)
        
        self.criterion = nn.BCELoss()
        
        self.G = Generator(noise_dim, self.embed_dim).to(self.device)
        self.D = Discriminator(self.embed_dim).to(self.device)