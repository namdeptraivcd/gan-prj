import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from src.model.base.cv.diffusion.utils.dataset import TextImageDataset
from src.config.config import GloalConfig, DiffusionConfig


class DiffusionTrainer:
    def __init__(self, diffusion):
        self.device = GloalConfig.DEIVCE
        self.diffusion = diffusion
        
        self.dataset = TextImageDataset(DiffusionConfig.dataset_path, DiffusionConfig.image_size)
        self.loader = DataLoader(self.dataset, batch_size=DiffusionConfig.batch_size, shuffle=True)
        
        self.optim = Adam(diffusion.unet.parameters(), lr=DiffusionConfig.lr)
        
        # Create folder checkpoints
        os.makedirs("data/cv/checkpoints", exist_ok=True)
        self.best_loss = float("inf")
    
    def train_minibatch(self, imgs, captions):
        imgs = imgs.to(self.device)
        context = self.diffusion.text_encoder(captions).to(self.device)
        
        # sample random timesteps
        t = torch.randint(1, DiffusionConfig.timesteps, (imgs.size(0),)).to(self.device)

        # add noise to images
        noised = self.diffusion.noise_images(imgs, t)

        # predict denoised image
        pred = self.diffusion.unet(noised, t, context)

        # compute loss
        
        # Debug
        print("imgs:", imgs.shape)
        print("pred:", pred.shape)

        
        loss = ((pred - imgs) ** 2).mean()

        # backpropagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()
    
    def train_epoch(self, epoch):
        pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}")
        total_loss = 0.0
        
        for imgs, captions in pbar:
            loss = self.train_minibatch(imgs, captions)
            total_loss += loss
            pbar.set_postfix(loss=loss)

        return total_loss / len(self.loader)
    
    def fit(self):
        self.diffusion.unet.train()
        
        for epoch in range(GloalConfig.NUM_MAX_EPOCHS):
            avg_loss = self.train_epoch(epoch)
            print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, avg_loss)
                print(f"Saved best model at epoch {epoch+1} with loss {avg_loss:.4f}")
    
    def save_checkpoint(self, epoch, loss):
        """Save best model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "unet": self.diffusion.unet.state_dict(),
            "text_encoder": self.diffusion.text_encoder.state_dict(),
            "optimizer": self.optim.state_dict(),
        }
        torch.save(checkpoint, "data/cv/checkpoints/best_model.pt")

    def load_checkpoint(self, path="data/cv/checkpoints/best_model.pt"):
        """Load saved model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.diffusion.unet.load_state_dict(checkpoint["unet"])
        self.diffusion.text_encoder.load_state_dict(checkpoint["text_encoder"])
        self.optim.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}, loss {checkpoint['loss']:.4f}")