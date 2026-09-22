"""근거 충분성 Check + 추가 Retrieval (설계문서 7장/11장의 Evidence Check 분기).

판정은 LLM에 다시 묻지 않고 규칙 기반으로 한다. 각 에이전트 프롬프트가 근거가 없을 때
"확인 불가" 류의 문구를 남기도록 지시되어 있으므로, 그 표지를 세는 것만으로 결정적이고
비용 없이 판정할 수 있다.

부족으로 판정되면 해당 관점만 Query Transformation을 켠 재검색으로 다시 돌린다.
재검색은 최대 1회(MAX_RETRY)이며, 그 후에는 근거가 여전히 부족해도 종합 단계로 넘어간다.
"""
from agents.market import market_node
from agents.domain import domain_node
from agents.tech_research import tech_research_node

MAX_RETRY = 1

# 도메인 평가처럼 지표별로 서술하는 프롬프트는 5개 중 1개만 근거가 없어도 표지를 남긴다.
# 그 정도는 정상 범위로 보고, SW/HW를 합쳐 표지가 2개 이상일 때만 "부족"으로 판정한다.
MIN_MARKERS = 2

INSUFFICIENT_MARKERS = (
    "확인 불가",
    "발췌 없음",
    "원문 미확보",
    "근거가 부족",
    "근거 부족",
    "확인되지 않",
)


def _marker_count(text: str) -> int:
    return sum(text.count(m) for m in INSUFFICIENT_MARKERS)


def _weak_perspectives(state: dict) -> list[str]:
    """RAG를 쓰는 관점 중 근거가 부족한 것들의 노드 이름 목록."""
    weak = []

    tr = state.get("tech_research", {})
    tech_text = " ".join(
        str(v) for side in ("sw", "hw") for v in tr.get(side, {}).values()
    )
    if _marker_count(tech_text) >= MIN_MARKERS:
        weak.append("tech_research")

    for name, key in (("market", "market_result"), ("domain", "domain_result")):
        result = state.get(key, {})
        text = " ".join(str(result.get(side, "")) for side in ("sw", "hw"))
        if _marker_count(text) >= MIN_MARKERS:
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
