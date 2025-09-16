import torch
from src.model.gan.networks.gan import GAN
from src.model.gan.utils.utils import embeddings_to_text, prepare_embeddings
from src.model.vae.networks.vae import VAE


def main(): #TODO: Thêm attention mask
    # Đọc dữ liệu và tạo embedding
    input_embeddings = prepare_embeddings("src/data/Chinese.xlsx", column_name="Utterance")

    # Tạo model (GAN/VAE)
    model = VAE(input_embeddings)
    
    # Train 
    model.fit()
    
    if isinstance(model, GAN):

        # Sinh embedding giả
        noise = torch.randn(5, 100)
        fake_embeddings = model.G(noise)

        # Convert embedding giả sang văn bản
        fake_texts = embeddings_to_text(fake_embeddings)
        for t in fake_texts:
            print(t)
    
    elif isinstance(model, VAE):
        raise NotImplementedError("Model will be implemented soon")
    
    else:
        raise NotImplementedError("Model not be supported")


if __name__ == "__main__":
    main()
