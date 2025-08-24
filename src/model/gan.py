import torch
import torch.nn as nn
import torch.optim as optim
from src.model.generator import Generator
from src.model.discriminator import Discriminator
from src.config.config import Config
config = Config()


class GAN:
    def __init__(self, real_embeddings, noise_dim=100, lr=0.0002):
        self.device = config.DEIVCE
        self.noise_dim = noise_dim
        self.embed_dim = real_embeddings.size(1)
        self.real_embeddings = real_embeddings.to(self.device)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Networks
        self.G = Generator(noise_dim, self.embed_dim).to(self.device)
        self.D = Discriminator(self.embed_dim).to(self.device)
        
        # Optimers
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=lr)
    
    def train_epoch(self, real_embeddings, G, D, noise_dim, device, criterion, optimizer_D, optimizer_G, epoch, epochs):
        # Train Discriminator 
        z = torch.randn(real_embeddings.size(0), noise_dim).to(device)
        fake_embeddings = G(z).detach()
        real_labels = torch.ones(real_embeddings.size(0), 1).to(device)
        fake_labels = torch.zeros(real_embeddings.size(0), 1).to(device)

        outputs_real = D(real_embeddings)
        outputs_fake = D(fake_embeddings)

        d_loss_real = criterion(outputs_real, real_labels)
        d_loss_fake = criterion(outputs_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator 
        z = torch.randn(real_embeddings.size(0), noise_dim).to(device)
        fake_embeddings = G(z)
        outputs = D(fake_embeddings)

        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    
    def fit(self):
        for epoch in range(config.NUM_MAX_EPOCHS):
            self.train_epoch(self.real_embeddings, self.G, self.D, self.noise_dim, self.device, self.criterion, self.optimizer_D, self.optimizer_G, epoch, config.NUM_MAX_EPOCHS)
            
        