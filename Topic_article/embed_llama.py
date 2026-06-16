# embed_llama.py
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models
from dotenv import load_dotenv

load_dotenv() # .env 파일에서 토큰 로드

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def chunk_text_by_tokens(text, tokenizer, max_len=512, stride=64):
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, lens = [], []
    i = 0
    n = len(ids)
    if n == 0: return [""], [0]
    while i < n:
        piece = ids[i:i+max_len]
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))
        if i + max_len >= n: break
        i += max_len - stride
    lens = [len(tokenizer.encode(c, add_special_tokens=False)) for c in chunks]
    return chunks, lens

def main():
    df = pd.read_csv("clean_data.csv")
    docs = df['lemma_text'].astype(str).tolist()

    print("🤖 Llama-3.2 임베딩 모델 로드 중...")
    word_embedding_model = models.Transformer(
        model_name_or_path="meta-llama/Llama-3.2-1B",
        max_seq_length=512,
        model_args={"torch_dtype": torch.float32, "device_map": "auto", "trust_remote_code": True}
    )
    
    tokenizer = word_embedding_model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        word_embedding_model.auto_model.config.pad_token_id = tokenizer.eos_token_id

    pooling_model = models.Pooling(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=True, pooling_mode_max_tokens=True, pooling_mode_mean_tokens=True
    )
    embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    print("🚀 청크 단위 임베딩 생성 시작...")
    emb_list = []
    with torch.no_grad():
        for doc in tqdm(docs, desc="Encoding Chunks"):
            chunks, lens = chunk_text_by_tokens(doc, tokenizer, 512, 64)
            chunk_embs = embedding_model.encode(chunks, batch_size=16, convert_to_numpy=True, show_progress_bar=False)
            
            w = np.asarray(lens, dtype=np.float32)
            w = w / (w.sum() if w.sum() != 0 else 1.0)
            doc_emb = (chunk_embs * w[:, None]).sum(axis=0)
            emb_list.append(doc_emb)

    embeddings = np.vstack(emb_list)
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/saved_llama_embeddings.npy", embeddings)
    print("✅ 임베딩 저장 완료: embeddings/saved_llama_embeddings.npy")

if __name__ == "__main__":
    main()