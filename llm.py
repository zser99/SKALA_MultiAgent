"""LLM / Embedding 팩토리.

- Generator LLM: OpenAI (기본 gpt-4o-mini, .env로 교체 가능)
- Embedding: 오픈소스 다국어 임베딩 (기본 BAAI/bge-m3, 팀 최종 결정)
  -> 선정 이유: 영문 논문 + 한국어 질의 간 cross-lingual retrieval 성능이 좋고,
     최대 8192 토큰의 긴 문맥을 지원하며, Dense/Sparse/Multi-vector를 한 모델에서
     지원해 향후 Hybrid Search로 확장 가능 (성능·한국어·긴 문서·운영편의·확장성의 균형).
  -> 주의: langchain_huggingface.HuggingFaceEmbeddings는 sentence-transformers 경로로
     bge-m3의 Dense 임베딩만 사용한다. Sparse/Multi-vector(ColBERT)까지 쓰는 완전한
     Hybrid Search는 FlagEmbedding의 BGEM3FlagModel을 직접 붙여야 하며, 이는 팀
     선정 사유에서 언급한 "향후 확장" 범위로 남겨둔다.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

_llm = None
_embeddings = None


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    global _llm
    model = os.getenv("LLM_MODEL", "gpt-5-mini")
    if _llm is None or _llm.model_name != model:
        _llm = ChatOpenAI(model=model, temperature=temperature)
    return _llm


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        _embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings
