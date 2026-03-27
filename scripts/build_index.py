import os, glob, json, math
import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

CASES_DIR = "CaseFiles"  # your .txt folder
OUT_DIR = "output"
CHUNK_CHARS = 800
MODEL_ID = "intfloat/multilingual-e5-base"  # multilingual 768-d

def read_txt(dirpath):
    paths = sorted(glob.glob(os.path.join(dirpath, "**/*.txt"), recursive=True))
    data = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
        data.append((name, txt))
    return data

def chunk(text, n):
    if not text: return []
    out = []
    for i in range(0, len(text), n):
        out.append(text[i:i+n])
    return out

def build_corpus(cases):
    titles, texts = [], []
    for name, txt in cases:
        parts = chunk(txt, CHUNK_CHARS) or [txt]
        for j, c in enumerate(parts):
            titles.append(f"{name}~Chunk {j+1}")
            texts.append(c)
    return {"title": titles, "text": texts}

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts

def encode_passages(corpus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModel.from_pretrained(MODEL_ID).to(device)
    mdl.eval()
    X = []
    bs = 16
    for i in tqdm(range(0, len(corpus["text"]), bs)):
        batch = [f"passage: {t}" for t in corpus["text"][i:i+bs]]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs)
        emb = mean_pool(out.last_hidden_state, inputs["attention_mask"]).detach().cpu().numpy()
        # normalize for inner product search
        faiss.normalize_L2(emb)
        X.append(emb.astype("float32"))
    return np.concatenate(X, axis=0)

def build_ip_index(X):
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    return index

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = read_txt(CASES_DIR)
    corpus = build_corpus(cases)
    X = encode_passages(corpus)
    index = build_ip_index(X)
    with open(os.path.join(OUT_DIR, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    print("Saved corpus.json and index.faiss")