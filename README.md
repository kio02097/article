# 🧬 Embedding & Pooling Strategies for Downstream NLP Tasks

본 저장소는 자연어 처리(NLP) 분야에서 임베딩 모델(Encoder vs. Decoder)과 다양한 풀링 전략(CLS, Mean, Max, Dual Pooling)의 조합이 다운스트림 태스크(**토픽 모델링** 및 **RAG 프레임워크**)의 성능과 효율성에 미치는 영향을 실증적으로 분석한 연구 코드를 포함하고 있습니다.

---

## 📌 연구 요약 및 로드맵 (Overview)

본 프로젝트는 두 가지 핵심 서브 연구(Sub-projects)로 구성되어 있습니다. 아래 링크를 클릭하시면 각 연구의 상세 섹션으로 바로 이동합니다.

1. [📂 [Topic_article] BERTopic에서 임베딩 모델 및 풀링 전략에 따른 토픽 모델링 성능 비교 분석](#1-topic_article-bertopic-토픽-모델링-성능-비교)
2. [📂 [RAG_Korean] RAG 프레임워크에서 검색과 생성 성능의 관계 분석](#2-rag_korean-rag-프레임워크에서-검색과-생성-성능-관계-분석)

---

## 1. 📂 [Topic_article] BERTopic 토픽 모델링 성능 비교

### Ⅰ. 개요 및 파이프라인
기존 BERTopic에서 관습적으로 사용되던 Mean 풀링을 넘어, 모델 구조(S-BERT vs LLaMA)와 단일/듀얼 풀링 기법의 조합을 교차 평가하여 최적의 아키텍처 가이드라인을 제시합니다.
* **파이프라인**: 전처리(SpaCy) ➡️ 은닉 상태 추출 ➡️ 풀링 적용(단일 또는 Mean+Max Concat) ➡️ BERTopic(UMAP, HDBSCAN, c-TF-IDF)

### Ⅱ. 실험 환경
* **임베딩 모델**: `all-MiniLM-L6-v2` (S-BERT), `Llama-3.2-1B` (LLaMA)
* **데이터셋**: BBC News (2,225개), 20Newsgroups (18,100개), IMDB (50,000개)

### Ⅲ. 핵심 결과 (Key Findings)
* **S-BERT(인코더)**는 CLS 또는 Max/Dual 풀링에서 우수했고, **LLaMA(디코더)**는 Mean 풀링에서 가장 안정적이었습니다.
* **경량 모델인 S-BERT에 듀얼 풀링(Dual Pooling)을 적용하면, 무거운 LLaMA 모델을 사용한 기본 임베딩과 유사하거나 더 높은 수준의 토픽 일관성($C_v$)을 확보**할 수 있습니다.
* **연산 효율성 (IMDB 데이터셋 기준)**:
  
  | 임베딩 모델 | 단일 풀링 소요 시간 | 듀얼 풀링 소요 시간 | 하드웨어 환경 |
  | :--- | :---: | :---: | :--- |
  | **S-BERT** | 11분 59초 | 13분 40초 | L4 GPU |
  | **LLaMA** | 47분 50초 | 48분 47초 | A100 GPU |

---

## 2. 📂 [RAG_Korean] RAG 프레임워크에서 검색과 생성 성능 관계 분석

### Ⅰ. 개요
RAG(Retrieval-Augmented Generation) 시스템 구축 시 검색 단계의 성능 향상이 최종 생성 단계의 품질 향상으로 직결되지 않는 **'성능 불일치(Performance Inconsistency)'** 현상을 실증적으로 분석합니다.

### Ⅱ. 실험 환경
* **임베딩 모델**: S-BERT, BGE, Solon, Gemma (Pooling: CLS, Max, Mean)
* **데이터셋 & LLM**: BioASQ4 (질문 1,022개, 문헌 13,007개) / `gemma3:4b`
* **검색 문서 수 ($k$)**: 5, 10, 20

### Ⅲ. 핵심 결과 (Key Findings)
* **성능 비일관성 확인**: 검색 단계(Recall, nDCG) 최고 점수 조합이 최종 생성(ROUGE, BLEU) 최고 성능을 보장하지 않으므로 End-to-End 평가가 필수적입니다.
* **질문 유형 및 $k$값 최적화**:
  * **Factoid & List (단답/나열형)**: 구체적 정보가 중요하므로 문서 수 선호도가 높음 (**$k=20$** 최적).
  * **Yes/No (판단형)**: 문서 수가 많아지면 LLM에 노이즈로 작용하여 생성 품질이 저하됨 (**$k=5$ 또는 $k=10$** 최적).

---

## 💡 총평 및 결론 (Overall Conclusion)

두 연구는 공통적으로 **"Mean 풀링이 모든 문맥과 태스크에서 항상 최선은 아니다"**라는 점을 시사합니다. 

* **토픽 모델링** 환경에서는 경량 모델에 **듀얼 풀링(Dual Pooling)**을 결합하여 LLM급의 효율을 낼 수 있습니다.
* **RAG 프레임워크** 환경에서는 유입되는 **질문의 성격(단답형 vs 판단형)에 따라 검색 문서 수($k$)와 풀링 전략을 유연하게 조정하는 동적 검색 전략(Dynamic Retrieval)**이 고도화의 핵심입니다.
