"""이해관계자 분석에서 근거가 부족한 기술과 평가 기준을 한 번의 웹 검색으로 보완한다."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_tavily import TavilySearch


# 동일한 검색 설정의 도구를 재사용하도록 생성 결과를 보관한다.
@lru_cache(maxsize=4)
def _get_search_tool(max_results: int = 6, search_depth: str = "basic") -> TavilySearch:
    return TavilySearch(
        max_results=max_results,
        topic="general",
        search_depth=search_depth,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )


# 검색 응답을 제목·출처·본문·관련도 점수가 있는 목록으로 정리한다.
def _normalize_result(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []

    if isinstance(raw, dict):
        items = raw.get("results", [])
    elif isinstance(raw, list):
        items = raw
    else:
        return [{"title": "Tavily result", "url": "", "content": str(raw), "score": None}]

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score"),
            }
        )
    return normalized


# 보완 대상으로 지정된 기술과 평가 기준만 묶어 하나의 검색어를 만든다.
def build_missing_evidence_query(
    sw_title: str,
    hw_title: str,
    missing_criteria: list[dict[str, Any]],
    *,
    max_query_chars: int = 1800,
) -> str:
    titles = {"sw": sw_title, "hw": hw_title}
    targets: dict[str, list[str]] = {"sw": [], "hw": []}
    observations: list[str] = []
    for spec in missing_criteria:
        criterion = str(spec.get("criterion", "")).strip()
        sides = spec.get("missing_for", [])
        if not criterion or not isinstance(sides, list):
            continue
        valid_sides = [side for side in titles if side in sides]
        for side in valid_sides:
            if criterion not in targets[side]:
                targets[side].append(criterion)
        if valid_sides:
            observations.extend(str(item) for item in spec.get("items", []))

    chunks = [
        f'"{titles[side]}": {", ".join(criteria)}'
        for side, criteria in targets.items() if criteria
    ]
    if not chunks:
        return ""

    # 길이 제한 때문에 뒤쪽 기술이나 평가 기준이 잘리지 않도록 먼저 모두 넣는다.
    query = " | ".join(chunks) + " | KV cache data center inference evidence"
    if len(query) > max_query_chars:
        raise ValueError("전체 보완 대상의 검색어가 최대 글자 수를 초과했습니다.")
    for item in dict.fromkeys(observations):
        if len(query) + len(item) + 3 <= max_query_chars:
            query += f" | {item}"
    return query


# 보완 대상이 있을 때만 검색을 한 번 실행하고 정리된 결과를 반환한다.
def search_missing_evidence(
    sw_title: str,
    hw_title: str,
    missing_criteria: list[dict[str, Any]],
    *,
    max_results: int = 6,
    search_depth: str = "basic",
) -> dict[str, Any]:
    query = build_missing_evidence_query(sw_title, hw_title, missing_criteria)
    if not query:
        return {"query": "", "results": []}
    tool = _get_search_tool(max_results=max_results, search_depth=search_depth)
    raw = tool.invoke({"query": query})
    return {
        "query": query,
        "results": _normalize_result(raw),
    }
