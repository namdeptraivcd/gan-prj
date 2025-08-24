import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ======================
# 1. Đọc file Excel và tạo embedding từ GPT2 Chinese
# ======================
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


# ======================
# 2. Generate văn bản từ embeddings (trick: convert embedding -> text seed)
# ======================
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


if __name__ == "__main__":
    print(">>> Đọc dữ liệu và sinh embeddings...")
    embeddings = read_excel("src/data/Chinese.xlsx", sheet_name=0, column_name="Utterance")
    print("Embedding shape:", len(embeddings), "câu")

    print(">>> Sinh văn bản từ GPT-2 Chinese...")
    fake_texts = embeddings_to_text(embeddings)
    for t in fake_texts:
        print(">>>", t)
