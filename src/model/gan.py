import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from src.data.utils import read_excel, embeddings_to_text
from transformers import BertTokenizer, BertModel
from src.model.network.generator import Generator
from src.model.network.discriminator import Discriminator


def prepare_embeddings(path, column_name="Utterance"):
    texts = read_excel(path, column_name=column_name)

    # Nếu read_excel trả về list[str], cần encode bằng BERT
    if isinstance(texts[0], str):
        print(">>> Dữ liệu là text, tạo embedding bằng BERT...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()

        embeddings = []
        for t in texts:
            inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)  # mean pooling
            embeddings.append(emb.squeeze(0).numpy())

        embeddings = np.array(embeddings, dtype=np.float32)

    else:
        # Nếu đã là số thì ép sang numpy float32
        embeddings = np.array(texts, dtype=np.float32)

    return torch.tensor(embeddings)



def train_gan(real_embeddings, noise_dim=100, epochs=200, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = real_embeddings.size(1)

    G = Generator(noise_dim, embed_dim).to(device)
    D = Discriminator(embed_dim).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    real_embeddings = real_embeddings.to(device)

    for epoch in range(epochs):
        # ---- Train Discriminator ----
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

        # ---- Train Generator ----
        z = torch.randn(real_embeddings.size(0), noise_dim).to(device)
        fake_embeddings = G(z)
        outputs = D(fake_embeddings)

        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    return G