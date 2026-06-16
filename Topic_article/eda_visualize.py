# eda_visualize.py
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer

def generate_wordcloud(df):
    STOPWORDS.update(['say', '_', 'maxaxaxaxaxaxaxaxaxaxaxaxaxaxax', 'mg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9vg9v'])
    
    for label in sorted(df['label'].unique()):
        text = " ".join(df.loc[df['label'] == label, 'lemma_text'])
        wc = WordCloud(width=800, height=800, max_words=200, stopwords=STOPWORDS, background_color="white").generate(text)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f"Label {label} WordCloud", fontsize=15)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"wordcloud_label_{label}.png")
        print(f"💾 Label {label} 워드클라우드 저장 완료.")

if __name__ == "__main__":
    df = pd.read_csv("clean_data.csv")
    print("📊 라벨별 문자 수 통계:")
    df['char_length'] = df['lemma_text'].str.len()
    print(df.groupby('label')['char_length'].describe())
    generate_wordcloud(df)