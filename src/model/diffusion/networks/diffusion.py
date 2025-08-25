import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from src.config.config import GloalConfig, DiffusionConfig


class Diffusion:
    def __init__(self, unet, text_encoder):
        self.device = GloalConfig.DEIVCE
        self.unet = unet.to(self.device)
        self.text_encoder = text_encoder.to(self.device)
        self.timesteps = DiffusionConfig.timesteps
    
    def noise_images(self, x, t):
        noise = torch.randn_like(x)
        t = t.view(-1, 1, 1, 1).float()
        return x + noise * (t / self.timesteps)
    
    def sample(self, prompt):
        self.unet.eval()
        self.text_encoder.eval()
        
        context = self.text_encoder([prompt])
        img = torch.randn(1, 3, DiffusionConfig.image_size, DiffusionConfig.image_size).to(self.device)
        
        for t in reversed(range(1, self.timesteps)):
            noise_pred = self.unet(img, torch.tensor([t]).to(self.device), context)
            img = img - noise_pred * (1.0 / self.timesteps)
        
        img = (img.clamp(-1, 1) + 1) / 2
        return to_pil_image(img[0].cpu())
    
    