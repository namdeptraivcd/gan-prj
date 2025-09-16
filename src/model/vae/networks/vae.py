from src.model.vae.networks.decoding_network import Decoder
from src.model.vae.networks.encoder import Encoder
from src.config.config import Config
from torch import optim
from torch import nn
import torch


cfg = Config()
class VAE:
    def __init__(self, input_embedding, latent_dim = 20, hidden_dim = 200):
        self.lr = 0.002
        self.device = cfg.DEIVCE
        self.epochs = cfg.NUM_MAX_EPOCHS
        self.batch_size = 32

        self.input_embedding = input_embedding
        input_dim = self.input_embedding.shape[1]
        output_dim = self.input_embedding.shape[1] # output is reconstructured input

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters())) # 
        self.loss = nn.MSELoss()


    def train_epoch(self):
        mean, logvar = self.encoder.forward(self.input_embedding)
        standard_deviation = torch.exp(0.5 * logvar)
        reconstructured_input = self.decoder.forward(mean, standard_deviation)
        
        #Loss
        KullBack_Leiber_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) #TODO: hiểu về KL Divergence
        reconstruction_loss = self.loss(reconstructured_input, self.input_embedding)
        self.l = reconstruction_loss + KullBack_Leiber_divergence
        
        #Train
        self.optimizer.zero_grad()
        self.l.backward()
        self.optimizer.step()

            

    def fit(self):
        for epoch in range (self.epochs):
            self.train_epoch()
            print(f"Epoch: {epoch+1}/{self.epochs} | Loss: {self.l.item():.4f}")