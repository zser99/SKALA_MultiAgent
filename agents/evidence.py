"""근거 부족 표지 Check + 추가 Retrieval (설계문서 7장/11장의 Evidence Check 분기).

판정은 LLM에 다시 묻지 않는 규칙 기반 검사다. 기술·시장·도메인 RAG 결과의 필수 값
누락과 검색 실패에 가까운 근거 부족 표지를 탐지한다. 이 검사는 출처의 정확성 자체를
검증하는 기능은 아니다. 단, 공개 자료로 확인되지 않는 운영·시장 지표를 분석 한계로
남기는 정상적인 "확인 불가" 표현은 재검색 실패로 보지 않는다.

부족으로 판정되면 해당 RAG 관점만 Query Transformation을 켠 재검색으로 다시 돌린다.
재검색은 최대 1회(MAX_RETRY)이며, 그 후에는 근거가 여전히 부족해도 종합 단계로 넘어간다.
이해관계자 결과는 자체 검색 후 ``결과 없음``으로 반환될 수 있는 정상 결과이므로 재검색
대상에 포함하지 않는다.
"""
from agents.market import market_node
from agents.domain import domain_node
from agents.tech_research import tech_research_node

MAX_RETRY = 1

# 원문·출처 자체가 없거나 검색 결과가 비어 있다는 표지는 한 번만 나타나도 재검색이 필요하다.
CRITICAL_MARKERS = (
    "발췌 없음",
    "검색된 발췌 없음",
    "원문 미확보",
    "출처 없음",
)

# 일부 지표의 단순한 "확인 불가"는 보고서 한계로 남길 수 있으므로 제외한다.
# 아래 표지는 검색이 실제로 실패했거나 입력 근거 자체가 비어 있음을 더 강하게 시사할 때만 쓴다.
SOFT_MARKERS = (
    "검색 실패",
    "검색 결과 없음",
    "직접 근거 없음",
)
MIN_SOFT_MARKERS = 2

# 도메인 평가는 5개 기준 x 2개 기술을 다루므로 일부 기준의 근거 누락은 최종 보고서의
# 분석 한계로 남긴다. 과반에 가까운 조합이 비어 있을 때만 검색 자체가 약하다고 보고 재검색한다.
MIN_DOMAIN_EVIDENCE_GAPS = 6


def _marker_count(value: object, markers: tuple[str, ...]) -> int:
    text = str(value)
    return sum(text.count(marker) for marker in markers)


def _is_missing(value: object) -> bool:
    """필수 결과가 비어 있거나 명시적으로 누락된 경우를 판별한다."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return not value
    return False


def _has_evidence_gap(value: object) -> bool:
    """치명적 표지 1개 또는 완화 표지 2개 이상을 근거 부족으로 본다."""
    return (
        _marker_count(value, CRITICAL_MARKERS) > 0
        or _marker_count(value, SOFT_MARKERS) >= MIN_SOFT_MARKERS
    )


def _domain_has_evidence_gap(result: dict) -> bool:
    """도메인 노드가 제공하는 구조화된 근거 누락 정보를 우선 사용한다."""
    analysis = result.get("domain_analysis")
    if isinstance(analysis, dict) and isinstance(analysis.get("evidence_gaps"), list):
        return len(analysis["evidence_gaps"]) >= MIN_DOMAIN_EVIDENCE_GAPS

    # 이전 형식의 결과와도 호환되도록 구조화된 정보가 없을 때만 문구를 검사한다.
    text = " ".join(str(result.get(side, "")) for side in ("sw", "hw"))
    return _has_evidence_gap(text)


def _gap_details(state: dict, weak: list[str]) -> str:
    """경고 메시지에 재검색 대상이 된 항목을 짧게 붙인다."""
    details = []
    if "market" in weak:
        market = state.get("market_result") or {}
        for side in ("sw", "hw"):
            for line in str(market.get(side, "")).splitlines():
                if _marker_count(line, CRITICAL_MARKERS + SOFT_MARKERS):
                    details.append(f"market.{side}.{line.split(':', 1)[0].strip()}")
    if "domain" in weak:
        analysis = (state.get("domain_result") or {}).get("domain_analysis") or {}
        for gap in analysis.get("evidence_gaps", []):
            if isinstance(gap, dict):
                details.append(f"domain.{gap.get('technology', '?')}.{gap.get('criterion', '?')}")
    return f" (부족 항목: {', '.join(details)})" if details else ""


def _weak_perspectives(state: dict) -> list[str]:
    """재검색이 필요한 RAG 관점의 노드 이름 목록.

    이해관계자 결과의 ``결과 없음``은 자체 검색을 마친 정상적인 분석 결과로 보고,
    이 함수의 검사 및 재검색 대상에서 제외한다.
    """
    weak = []

    raw_tr = state.get("tech_research", {})
    tr = raw_tr if isinstance(raw_tr, dict) else {}
    tech_missing = (
        not isinstance(raw_tr, dict)
        or any(_is_missing(tr.get(side)) for side in ("sw", "hw"))
    )
    tech_text = " ".join(
        str(value)
        for side in ("sw", "hw")
        for value in (
            tr.get(side, {}).values()
            if isinstance(tr.get(side), dict)
            else [tr.get(side, "")]
        )
    )
    if tech_missing or _has_evidence_gap(tech_text):
        weak.append("tech_research")

    for name, key in (("market", "market_result"), ("domain", "domain_result")):
        raw_result = state.get(key, {})
        result = raw_result if isinstance(raw_result, dict) else {}
        result_missing = (
            not isinstance(raw_result, dict)
            or any(_is_missing(result.get(side)) for side in ("sw", "hw"))
        )
        has_gap = (
            _domain_has_evidence_gap(result)
            if name == "domain"
            else _has_evidence_gap(
                " ".join(str(result.get(side, "")) for side in ("sw", "hw"))
            )
        )
        if result_missing or has_gap:
            weak.append(name)

    return weak


def evidence_check_node(state: dict) -> dict:
    weak = _weak_perspectives(state)
    warnings = list(state.get("warnings", []))
    retry_count = state.get("retry_count", 0)

    if weak:
        status = "재검색 수행" if retry_count < MAX_RETRY else "재검색 한도 소진, 그대로 종합"
        warnings.append(
            f"[evidence] 근거 부족 관점: {', '.join(weak)} -> {status}"
            f"{_gap_details(state, weak)}"
        )

    return {
        "evidence_sufficient": not weak,
        "retry_count": retry_count,
        "warnings": warnings,
    }


def evidence_router(state: dict) -> str:
    if state.get("evidence_sufficient") or state.get("retry_count", 0) >= MAX_RETRY:
        return "synthesis"
    return "re_retrieval"


def re_retrieval_node(state: dict) -> dict:
    """근거가 부족한 관점만 Query Transformation을 켜고 다시 검색·평가한다."""
    weak = set(_weak_perspectives(state))
    retry_state = {**state, "retry_count": state.get("retry_count", 0) + 1}

    out: dict = {"retry_count": retry_state["retry_count"]}
    if "tech_research" in weak:
        out.update(tech_research_node(retry_state))
    if "market" in weak:
        out.update(market_node(retry_state))
    if "domain" in weak:
        out.update(domain_node(retry_state))
    return out
