import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def prepare_rag_embeddings(data_dir="./data", persist_dir="./chroma_db", chunk_size=500, chunk_overlap=50):
    print("1. 문서 데이터 로드 중...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"'{data_dir}' 폴더가 생성되었습니다. 분석할 텍스트 파일을 넣어주세요.")
        return None
        
    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print(f"'{data_dir}' 폴더 안에 분석할 텍스트(.txt) 파일이 없습니다.")
        return None

    print(f"2. 텍스트 분할 중 (Chunk Size: {chunk_size})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    print(f"총 {len(docs)}개의 텍스트 청크가 생성되었습니다.")

    print("3. HuggingFace 임베딩 모델 로드 중...")
    model_name = "jhgan/ko-sroberta-multitask"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'} # GPU 사용 시 'cuda'
    )
    
    print("4. 크로마(Chroma) 벡터 데이터베이스 빌드 중...")
    # 데이터를 Chroma DB에 저장하고 지정한 디렉토리에 로컬 저장(상태 유지)합니다.
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Chroma DB가 '{persist_dir}' 경로에 성공적으로 저장되었습니다.")
    
    return vector_store

if __name__ == "__main__":
    prepare_rag_embeddings()