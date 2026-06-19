from chromadb import PersistentClient

def get_chroma_client(persist_dir: str):
    """지정된 경로의 Chroma PersistentClient 반환"""
    return PersistentClient(path=persist_dir)

def list_collections(client):
    """현재 DB에 존재하는 모든 컬렉션 이름 출력 및 리스트 반환"""
    collections = client.list_collections()
    collection_names = [c.name for c in collections]
    print("📚 존재하는 Collection 목록:")
    for name in collection_names:
        print(f" - {name}")
    return collection_names