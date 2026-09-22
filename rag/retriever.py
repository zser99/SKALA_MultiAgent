"""벡터스토어 검색.

- transform=False : Baseline (Dense Retrieval + Top-K)
- transform=True  : Query Transformation 적용 후 RRF(Reciprocal Rank Fusion)로 융합

두 경로 모두 `search()`를 거치므로, 에이전트와 평가 스크립트(rag/evaluate.py)가
완전히 동일한 검색 로직을 공유한다.
"""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.query_transform import transform_query

RRF_K = 60


def _doc_key(d: Document) -> tuple:
    return (d.metadata.get("source_file"), d.metadata.get("page"), d.page_content[:80])


def _rrf_fuse(ranked_lists: list[list[Document]], k: int) -> list[Document]:
    scores: dict[tuple, float] = {}
    docs: dict[tuple, Document] = {}
    for ranked in ranked_lists:
        for rank, d in enumerate(ranked):
            key = _doc_key(d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            docs.setdefault(key, d)
    top = sorted(scores, key=lambda key: scores[key], reverse=True)[:k]
    return [docs[key] for key in top]


def search(
    vectorstore: FAISS,
    query: str,
    k: int = 4,
    transform: bool = False,
    title: str = "",
) -> list[Document]:
    if vectorstore is None:
        return []
    if not transform:
        return vectorstore.similarity_search(query, k=k)

    queries = transform_query(query, title=title)
    ranked_lists = [vectorstore.similarity_search(q, k=k) for q in queries]
    return _rrf_fuse(ranked_lists, k)


def retrieve_context(
    vectorstore: FAISS,
    query: str,
    k: int = 4,
    transform: bool = False,
    title: str = "",
) -> str:
    """검색 결과를 프롬프트에 넣기 좋은 문자열로 반환."""
    docs = search(vectorstore, query, k=k, transform=transform, title=title)
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        parts.append(f"[p.{page}] {d.page_content.strip()}")
    return "\n\n".join(parts)
