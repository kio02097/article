import os

# 디렉터리 경로 설정
CHROMA_BGE_DIR = "C:/Users/USER/Desktop/RAG_Korean/chroma_db_bge"
RESULTS_DIR = "./results"
SUMMARY_DIR = "./summary"

# LLM 및 프롬프트 설정 (기본값 예시)
LLM_MODEL = "ollama/llama3"
PROMPT_PATH = "./prompts/rag_prompt.txt"

# 폴더 자동 생성
for d in [RESULTS_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)