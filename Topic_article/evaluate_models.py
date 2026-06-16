import os
import random
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from wordcloud import STOPWORDS  # ◀ [수정] 필수 임포트 추가

# 0) Seed 고정
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 1) 데이터 로드 ◀ [수정] 전처리된 CSV 파일을 읽어오도록 변경
if not os.path.exists("clean_data.csv"):
    raise FileNotFoundError("⚠️ 'clean_data.csv' 파일이 없습니다! preprocess.py를 먼저 실행해주세요.")

df_all = pd.read_csv("clean_data.csv")
# NaN 값이 있을 경우를 대비해 문자열 변환 및 정제
docs = df_all['lemma_text'].dropna().astype(str).tolist()
tokenized_docs = [doc.split() for doc in docs]

# 2) 불용어, 벡터라이저, 사전 준비
vectorizer_model = CountVectorizer(
    stop_words=list(STOPWORDS),
    ngram_range=(1,1),
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b"
)
dictionary = corpora.Dictionary(tokenized_docs)

# 3) 저장된 임베딩 로드
if not os.path.exists("embeddings/saved_llama_embeddings.npy"):
    raise FileNotFoundError("⚠️ 임베딩 파일이 없습니다! embed_llama.py를 먼저 실행해주세요.")

embeddings = np.load("embeddings/saved_llama_embeddings.npy")
assert embeddings.shape[0] == len(docs), f"임베딩 개수({embeddings.shape[0]})와 문서 개수({len(docs)})가 불일치합니다!"

# 4) nr_topics 평가 루프
results = []
for n in tqdm(range(2, 25, 2), desc="Evaluating nr_topics"):
    model_n = BERTopic(
        embedding_model=None,       # ★ precomputed embeddings 사용
        vectorizer_model=vectorizer_model,
        min_topic_size=10,
        nr_topics=n,
        calculate_probabilities=False
    )
    topics_n, _ = model_n.fit_transform(docs, embeddings=embeddings)

    # 토픽 수 계산
    info = model_n.get_topic_info()
    n_clusters = info[info.Topic != -1].shape[0]

    # coherence & diversity 계산
    topics = model_n.get_topics()
    tids   = sorted([t for t in topics if t != -1])
    if not tids:
        cv = cn = div = 0.0
    else:
        words = [[w for w,_ in topics[t]] for t in tids]
        cm_cv = CoherenceModel(topics=words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
        cm_cn = CoherenceModel(topics=words, texts=tokenized_docs, dictionary=dictionary, coherence='c_npmi')
        cv = cm_cv.get_coherence()
        cn = cm_cn.get_coherence()
        all_top = [w for tw in words for w in tw[:10]]
        div = len(set(all_top)) / (10 * len(words))

    results.append({
        'nr_topics':        n,
        'n_clusters':       n_clusters,
        'coherence_c_v':    cv,
        'coherence_c_npmi': cn,
        'topic_diversity':  div
    })

# 5) 결과 정리 및 최적값 출력
df_res = pd.DataFrame(results)
print("\n=== 전체 결과 ===")
print(df_res)

mean_values = df_res.mean(numeric_only=True)
mean_row = pd.DataFrame(mean_values).T
mean_row.index = ['mean']
df_res = pd.concat([df_res, mean_row])

print("\n=== 평균 포함 결과 ===")
print(df_res)

valid = df_res.drop(index='mean')
best_cv  = valid['coherence_c_v'].idxmax()
best_cn  = valid['coherence_c_npmi'].idxmax()
best_div = valid['topic_diversity'].idxmax()

print("\n=== 최적의 파라미터 ===")
print(f"▶ coherence_c_v 최고: nr_topics = {int(valid.loc[best_cv,'nr_topics'])}, 값 = {valid.loc[best_cv,'coherence_c_v']:.4f}")
print(f"▶ coherence_c_npmi 최고: nr_topics = {int(valid_df.loc[best_cn,'nr_topics'] if 'valid_df' in locals() else valid.loc[best_cn,'nr_topics'])}, 값 = {valid.loc[best_cn,'coherence_c_npmi']:.4f}")
print(f"▶ topic_diversity 최고: nr_topics = {int(valid.loc[best_div,'nr_topics'])}, 값 = {valid.loc[best_div,'topic_diversity']:.4f}")