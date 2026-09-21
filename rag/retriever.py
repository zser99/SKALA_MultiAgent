"""벡터스토어에서 질의 관련 컨텍스트를 뽑아 프롬프트에 넣기 좋은 문자열로 반환."""
from langchain_chroma import Chroma


def retrieve_context(vectorstore: Chroma, query: str, k: int = 4) -> str:
    if vectorstore is None:
        return ""
    docs = vectorstore.similarity_search(query, k=k)
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        parts.append(f"[p.{page}] {d.page_content.strip()}")
    return "\n\n".join(parts)
