import torch
import torch.optim as optim
from src.config.config import GloalConfig, GANConfig


class GANTrainer:
    def __init__(self, gan, lr=GANConfig.lr):
        self.gan = gan
        self.device = GloalConfig.DEIVCE
        self.optimizer_G = optim.Adam(self.gan.G.parameters(), lr=lr)
        self.optimizer_D = optim.Adam(self.gan.D.parameters(), lr=lr)

    def train_epoch(self, epoch, epochs):
        batch_size = self.gan.real_embeddings.size(0)
        noise_dim = GANConfig.noise_dim

        # Train Discriminator
        z = torch.randn(batch_size, noise_dim).to(self.deivce)
        fake_embeddings = self.gan.G(z).detach()
        real_labels = torch.ones(batch_size, 1).to(self.deivce)
        fake_labels = torch.zeros(batch_size, 1).to(self.deivce)

        outputs_real = self.gan.D(self.gan.real_embeddings)
        outputs_fake = self.gan.D(fake_embeddings)

        d_loss_real = self.gan.criterion(outputs_real, real_labels)
        d_loss_fake = self.gan.criterion(outputs_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, noise_dim).to(self.deivce)
        fake_embeddings = self.gan.G(z)
        outputs = self.gan.D(fake_embeddings)
        g_loss = self.gan.criterion(outputs, real_labels)

        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    def fit(self, num_epochs=GloalConfig.NUM_MAX_EPOCHS):
        for epoch in range(num_epochs):
            self.train_epoch(epoch, num_epochs)