"""선정된 후보 논문 PDF -> 청크 분할 -> 임베딩 -> Chroma 벡터스토어.

총 200페이지 한도(과제 조건)를 넘지 않도록, 로딩 시 페이지 수를 세어 경고한다.
PDF가 data/ 에 없으면 예외 대신 warning을 남기고 None을 반환해 그래프가 계속
진행될 수 있게 한다 (기술 조사 에이전트가 summary 필드로 폴백).
"""
import os
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from llm import get_embeddings

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
MAX_TOTAL_PAGES = 200


def _load_pdf(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None, f"[RAG] data/{filename} 파일을 찾을 수 없어 원문 검색 없이 진행합니다."
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages, None


def build_vectorstore(candidate_id: str, filename: str) -> tuple[Optional[Chroma], Optional[str]]:
    """단일 후보 논문에 대한 벡터스토어를 만들거나, 이미 있으면 재사용한다."""
    collection = f"cand_{candidate_id}"
    persist_path = os.path.join(PERSIST_DIR, collection)

    if os.path.exists(persist_path) and os.listdir(persist_path):
        vs = Chroma(
            collection_name=collection,
            embedding_function=get_embeddings(),
            persist_directory=persist_path,
        )
        return vs, None

    pages, warning = _load_pdf(filename)
    if pages is None:
        # PDF도 없고 기존 벡터스토어도 없음 -> 임베딩 모델 로드 자체를 건너뛴다.
        return None, warning

    embeddings = get_embeddings()

    if len(pages) > MAX_TOTAL_PAGES:
        warning = (
            f"[RAG] {filename} 이 {len(pages)}페이지로 과제 한도(200p)를 초과합니다. "
            f"앞 {MAX_TOTAL_PAGES}페이지만 사용합니다."
        )
        pages = pages[:MAX_TOTAL_PAGES]

    # bge-m3는 최대 8192 토큰(약 5000~6000자)까지 지원하므로, 문단 단위 맥락이
    # 끊기지 않도록 기존 e5-small 기준(800자)보다 넉넉한 청크 크기를 사용한다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    for c in chunks:
        c.metadata["source_id"] = candidate_id
        c.metadata["source_file"] = filename

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=persist_path,
    )
    return vs, warning
