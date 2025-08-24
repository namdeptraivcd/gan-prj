import torch
from src.model.gan import GAN
from src.utils.utils import embeddings_to_text, prepare_embeddings

def main():
    # Đọc dữ liệu và tạo embedding
    real_embeddings = prepare_embeddings("src/data/Chinese.xlsx", column_name="Utterance")

    model = GAN(real_embeddings=real_embeddings)
    
    # Train GAN
    model.fit()

    # Sinh embedding giả
    noise = torch.randn(5, 100)
    fake_embeddings = model.G(noise)

    # Convert embedding giả sang văn bản
    fake_texts = embeddings_to_text(fake_embeddings)
    for t in fake_texts:
        print(t)


if __name__ == "__main__":
    main()
