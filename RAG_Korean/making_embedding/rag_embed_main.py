import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rag_embed import prepare_rag_embeddings

def run_rag_pipeline():
    persist_dir = "./chroma_db"
    
    # 만약 이미 빌드된 크로마 DB 폴더가 없다면 새로 생성합니다.
    if not os.path.exists(persist_dir):
        vector_store = prepare_rag_embeddings(data_dir="./data", persist_dir=persist_dir)
        if not vector_store:
            return
    else:
        print("기존에 생성된 Chroma DB를 로드합니다...")
        model_name = "jhgan/ko-sroberta-multitask"
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print("\n" + "="*40)
    print("🚀 Chroma DB 기반 RAG 검색 준비 완료!")
    print("="*40)
    
    while True:
        query = input("\n🔍 검색할 질문을 입력하세요 (종료하려면 'q' 입력): ")
        if query.lower() == 'q':
            break
            
        print(f" 질문: {query}")
        print(" Chroma DB에서 유사도 검색 중...")
        
        # 크로마 DB에서 질문과 가장 유사한 상위 3개 청크 검색
        related_docs = vector_store.similarity_search(query, k=3)
        
        print("\n📌 관련도가 높은 본문 내용:")
        for i, doc in enumerate(related_docs, 1):
            print(f"\n[{i}번째 참조 문단] (출처: {doc.metadata.get('source', '알 수 없음')})")
            print(doc.page_content.strip())

if __name__ == "__main__":
    run_rag_pipeline()