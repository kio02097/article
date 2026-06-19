from config import CHROMA_BGE_DIR
from evaluator import RAGEvaluator
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    print("=== RAG Evaluation Framework 실행 ===")
    
    # 1. 사용할 임베딩 모델 로드 (HuggingFace 예시 - 필요시 모델명 변경)
    print("🤖 임베딩 모델 로드 중...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # 2. 평가기 객체 생성
    evaluator = RAGEvaluator(
        dataset_csv="RAG-Evaluation-Dataset-KO.csv", # 준비하신 데이터셋 파일명
        collection_name="bge_rag_evaluation_fium_500_cls", # 평가할 Chroma 컬렉션 이름
        k=5,
        embedding_function=embedding_model,
        chroma_dir=CHROMA_BGE_DIR,
        do_llm=True # Ollama 연동 및 답변 생성을 원치 않으시면 False로 변경 가능
    )
    
    # 3. 평가 실행
    evaluator.run_evaluation()
    print("✅ 모든 평가 파이프라인이 성공적으로 종료되었습니다.")

if __name__ == "__main__":
    main()