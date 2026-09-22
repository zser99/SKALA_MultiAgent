"""Query Transformation — Baseline(Dense Top-K) 대비 검색 개선 전략.

Baseline의 약점 두 가지를 겨냥한다.
1) 질의는 한국어인데 문서는 영문 논문이라 어휘가 직접 겹치지 않는다.
2) 한 질의에 여러 측면(메모리/지연/처리량 등)이 섞이면 단일 임베딩이 한 측면만 대표한다.

따라서 원 질의를 논문 어휘 중심의 영어 하위 질의 여러 개로 바꾸고, 각각 검색한 결과를
RRF로 융합한다. LLM 호출이 실패하면 원 질의만 돌려주어 Baseline으로 자연스럽게 폴백한다.
"""
from llm import get_llm
from agents.utils import parse_json_response

_PROMPT = """당신은 RAG 검색 질의 변환기입니다. 아래 한국어 질의를 영문 기술 논문 검색에
적합한 영어 질의 {n}개로 변환하세요.

규칙:
- 논문 본문에 실제로 등장할 법한 기술 용어/표현을 사용하세요 (일반어 풀어쓰기 금지).
- 원 질의에 여러 측면이 섞여 있으면 측면별로 나누어 서로 다른 질의로 만드세요.
- 각 질의는 한 문장 또는 명사구로 짧게 작성하세요.

[대상 문서] {title}
[원 질의] {query}

다음 JSON 형식으로만 답하세요:
{{"queries": ["...", "...", "..."]}}"""


def transform_query(query: str, title: str = "", n: int = 3) -> list[str]:
    """원 질의 + 변환된 영어 질의들을 반환한다 (원 질의는 항상 맨 앞에 유지)."""
    prompt = _PROMPT.format(n=n, title=title or "(미지정)", query=query)
    try:
        resp = get_llm(temperature=0).invoke(prompt)
        variants = parse_json_response(resp.content).get("queries", [])
    except Exception:
        return [query]

    out = [query]
    for v in variants:
        if isinstance(v, str) and v.strip() and v.strip() not in out:
            out.append(v.strip())
    return out
