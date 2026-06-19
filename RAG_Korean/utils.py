import os
import ast
import unicodedata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_document_list(documents):
    """원본 딕셔너리 리스트를 LangChain Document 객체 리스트로 변환"""
    return [
        Document(
            page_content=d["text"],
            metadata={
                "pmid": str(d.get("pmid", "")),
                "title": d.get("title", ""),
                "id": d.get("id", i),
            }
        )
        for i, d in enumerate(documents)
    ]

def split_documents(document_list, chunk_size=500, chunk_overlap=100):
    """문서를 지정된 크기(청크)로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(document_list)
    print(f"📄 총 {len(splits)}개의 텍스트 청크가 생성되었습니다.")
    return splits

def load_prompt_file(prompt_path: str):
    """프롬프트 파일에서 SYSTEM 및 USER 메시지를 분리하여 로드"""
    if not os.path.exists(prompt_path):
        return "", ""
    with open(prompt_path, "r", encoding="utf-8") as f:
        text = f.read()
    if "[SYSTEM]" in text and "[USER]" in text:
        sys_part = text.split("[SYSTEM]", 1)[1].split("[USER]", 1)[0].strip()
        user_part = text.split("[USER]", 1)[1].strip()
        return sys_part, user_part
    return "", text.strip()