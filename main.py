import torch
import argparse

from src.config.config import DiffusionConfig

from src.model.base.nlp.gan.networks.gan import GAN
from src.model.base.nlp.gan.train.trainer import GANTrainer
from src.model.base.nlp.gan.utils.utils import embeddings_to_text, prepare_embeddings

from src.model.base.nlp.vae.networks.vae import VAE

from src.model.base.cv.diffusion.networks.diffusion import Diffusion
from src.model.base.cv.diffusion.train.trainer import DiffusionTrainer
from src.model.base.cv.diffusion.networks.diffusion import Diffusion
from src.model.base.cv.diffusion.networks.unet import UNet
from src.model.base.cv.diffusion.networks.text_encoder import TextEncoder


def main():
    model_type = "Diffusion"

    if model_type == "GAN":
        # Đọc dữ liệu và tạo embedding
        real_embeddings = prepare_embeddings("data/nlp/Chinese.xlsx", column_name="Utterance")

        # Tạo model (GAN/VAE)
        model = GAN(real_embeddings=real_embeddings)
    
        trainer = GANTrainer(model)
    
        # Train 
        trainer.fit()
        
        # Sinh embedding giả
        noise = torch.randn(5, 100)
        fake_embeddings = model.G(noise)

        # Convert embedding giả sang văn bản
        fake_texts = embeddings_to_text(fake_embeddings)
        for t in fake_texts:
            print(t)
    elif model_type == "VAE":
        raise NotImplementedError("Model will be implemented soon")
    elif model_type == "Diffusion":
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", action="store_true")
        parser.add_argument("--sample", action="store_true")
        parser.add_argument("--prompt", type=str, default="A cat")
        args = parser.parse_args()

        # init models
        text_encoder = TextEncoder()
        unet = UNet()
        diffusion = Diffusion(unet, text_encoder)

        if args.train:
            trainer = DiffusionTrainer(diffusion)
            trainer.fit()
        elif args.sample:
            trainer = DiffusionTrainer(diffusion)
            trainer.load_checkpoint("data/cv/checkpoints/best_model.pt")  # Load best model
            img = diffusion.sample(args.prompt)
            img.save("data/cv/generated.png")
            print("Image saved to data/cv/generated.png")
    else:
        raise NotImplementedError("Model not be supported")
        

if __name__ == "__main__":
    main()
