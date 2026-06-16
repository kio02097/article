# preprocess.py
import re
import pandas as pd
import nltk
import spacy
from datasets import load_dataset, concatenate_datasets
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def tokenize_english_text(text):
    text = re.sub(r'[^A-Za-z0-9\w\s,\.?!]', '', text)
    tokens = text.lower().split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

def lemmatize_spacy(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def load_and_preprocess():
    print("📦 데이터셋 로드 중...")
    ds = load_dataset("SetFit/20_newsgroups")
    all_ds = concatenate_datasets([ds["train"], ds["test"]])
    df_all = all_ds.to_pandas()
    
    print("✨ 텍스트 정제 및 표제어 추출 중 (시간이 소요될 수 있습니다)...")
    df_all['clean_text'] = df_all['text'].apply(tokenize_english_text)
    df_all['lemma_text'] = df_all['clean_text'].apply(lemmatize_spacy)
    
    # 빈 문서 및 너무 짧은 문서 필터링
    df_all = df_all[df_all['lemma_text'].str.len() != 0]
    df_all['length'] = df_all['lemma_text'].str.split().map(len)
    df_all = df_all[df_all['length'] >= 5]
    
    df_all.to_csv("clean_data.csv", index=False)
    print("✅ 전처리 완료! 'clean_data.csv'로 저장되었습니다.")

if __name__ == "__main__":
    load_and_preprocess()