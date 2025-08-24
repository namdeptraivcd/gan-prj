import torch
from src.model import gan as model



if __name__ == "__main__":
    print(">>> Đọc dữ liệu và tạo embedding...")
    real_embeddings = model.prepare_embeddings("src/data/Chinese.xlsx", column_name="Utterance")

    print(">>> Train GAN...")
    G = model.train_gan(real_embeddings)

    print(">>> Sinh embedding giả...")
    noise = torch.randn(5, 100)
    fake_embeddings = G(noise)

    print(">>> Convert embedding giả sang văn bản...")
    fake_texts = model.embeddings_to_text(fake_embeddings)
    for t in fake_texts:
        print(">>>", t)
