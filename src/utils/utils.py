import numpy as np
import pandas as pd
import torch
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, 
                          BertTokenizer, BertModel)


# Đọc file Excel và tạo embedding từ GPT2 Chinese
def read_excel(file_path, sheet_name=0, column_name="Utterance"):
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = AutoModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")  # <-- dùng AutoModel để lấy hidden states
    model.eval()

    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=100)  # đọc 10000 dòng
    sentences = df[column_name].dropna().tolist()

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_dim)
        embeddings.append(cls_embedding.squeeze().numpy())  # (hidden_dim,)
    return embeddings


# Generate văn bản từ embeddings (trick: convert embedding -> text seed)
def embeddings_to_text(embeddings, max_length=50):
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model.eval()

    texts = []
    for _ in embeddings[:5]:  # chỉ generate 5 câu thử
        input_ids = tokenizer.encode("今天天气", return_tensors="pt")  # seed Chinese
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        texts.append(text)
    return texts


def prepare_embeddings(path, column_name="Utterance"):
    texts = read_excel(path, column_name=column_name)

    # Nếu read_excel trả về list[str], cần encode bằng BERT
    if isinstance(texts[0], str):
        print("Dữ liệu là text, tạo embedding bằng BERT...")
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
