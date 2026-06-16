# embed_sbert.py
import os
import random
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models

# 0) Seed 고정[cite: 1]
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def main():
    # 1) 데이터 로드 (전처리된 데이터 사용)
    if not os.path.exists("clean_data.csv"):
        raise FileNotFoundError("⚠️ 'clean_data.csv' 파일이 없습니다! preprocess.py를 먼저 실행해주세요.")
        
    df_all = pd.read_csv("clean_data.csv")
    docs = df_all['lemma_text'].dropna().astype(str).tolist()

    print("🤖 SBERT 임베딩 모델(all-MiniLM-L6-v2) 로드 중...[cite: 1]")
    # (1) 모델 + Pooling 구성[cite: 1]
    word_embedding_model = models.Transformer("sentence-transformers/all-MiniLM-L6-v2")[cite: 1]
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_max_tokens=True,
        pooling_mode_mean_tokens=True
    )[cite: 1]
    embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])[cite: 1]

    # (2) 임베딩 생성[cite: 1]
    print("🚀 SBERT 임베딩 생성 시작...")
    embeddings = embedding_model.encode(docs, batch_size=32, show_progress_bar=True, convert_to_numpy=True)[cite: 1]

    # (3) 저장[cite: 1]
    os.makedirs("embeddings", exist_ok=True)
    save_path = "embeddings/saved_sbert_embeddings.npy"
    np.save(save_path, embeddings)
    print(f"✅ SBERT 임베딩 저장 완료: {save_path}")

if __name__ == "__main__":
    main()