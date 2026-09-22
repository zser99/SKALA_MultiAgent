"""근거 부족 표지 Check + 추가 Retrieval (설계문서 7장/11장의 Evidence Check 분기).

판정은 LLM에 다시 묻지 않는 규칙 기반 검사다. 기술·시장·도메인 RAG 결과의 필수 값
누락과 ``확인 불가`` 류의 근거 부족 표지를 탐지한다. 이 검사는 출처의 정확성 자체를
검증하는 기능은 아니다.

부족으로 판정되면 해당 RAG 관점만 Query Transformation을 켠 재검색으로 다시 돌린다.
재검색은 최대 1회(MAX_RETRY)이며, 그 후에는 근거가 여전히 부족해도 종합 단계로 넘어간다.
이해관계자 결과는 자체 검색 후 ``결과 없음``으로 반환될 수 있는 정상 결과이므로 재검색
대상에 포함하지 않는다.
"""
from agents.market import market_node
from agents.domain import domain_node
from agents.tech_research import tech_research_node

MAX_RETRY = 1

# 원문·출처 자체가 없다는 표지는 한 번만 나타나도 재검색이 필요하다.
CRITICAL_MARKERS = (
    "확인 불가",
    "발췌 없음",
    "원문 미확보",
    "출처 없음",
)

# 도메인 평가처럼 지표별로 서술하는 결과에서는 일부 지표의 근거 부족이 있을 수 있다.
# 같은 결과 안에서 완화 표지가 두 번 이상 나타날 때 재검색 대상으로 판단한다.
SOFT_MARKERS = (
    "근거가 부족",
    "근거 부족",
    "확인되지 않",
)
MIN_SOFT_MARKERS = 2


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
        text = " ".join(str(result.get(side, "")) for side in ("sw", "hw"))
        result_missing = (
            not isinstance(raw_result, dict)
            or any(_is_missing(result.get(side)) for side in ("sw", "hw"))
        )
        if result_missing or _has_evidence_gap(text):
            weak.append(name)

    return weak


def evidence_check_node(state: dict) -> dict:
    weak = _weak_perspectives(state)
    warnings = list(state.get("warnings", []))
    retry_count = state.get("retry_count", 0)

    if weak:
        status = "재검색 수행" if retry_count < MAX_RETRY else "재검색 한도 소진, 그대로 종합"
        warnings.append(f"[evidence] 근거 부족 관점: {', '.join(weak)} -> {status}")

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
