"""보고서 생성 에이전트(RAG: X).
앞선 Agent가 State에 저장한 분석 결과만 사용해 보고서 본문을 생성한다.
REFERENCE는 LLM이 아닌 코드가 생성한다.
"""
import json
import re
from agents.references import (
    audit_sources,
    collect_sources,
    find_orphan_citations,
    render_references,
)
from agents.utils import load_prompt
from llm import get_llm

_REQUIRED_REPORT_HEADINGS = (
    "## 0. SUMMARY",
    "## 1. 분석 배경",
    "## 2. 대상 기술 선정",
    "### 2.1 SW 기술",
    "### 2.2 HW 기술",
    "### 2.3 선정 근거",
    "## 3. 기술 개요",
    "### 3.1 SW 기술 원리 및 한계",
    "### 3.2 HW 기술 원리 및 한계",
    "## 4. 관점별 평가",
    "### 4.1 기술 성숙도",
    "### 4.2 시장성",
    "### 4.3 이해관계자",
    "### 4.4 도메인 적용성",
    "## 5. 관점 간 비교 및 시사점",
    "### 5.1 일치하는 평가",
    "### 5.2 충돌하는 평가",
    "### 5.3 시사점",
    "## 6. 분석 한계",
    "### 6.1 공개 정보 기반 추정의 한계",
    "### 6.2 확증편향 방지 방법",
)


_REQUIRED_STATE_KEYS = (
    "selection_rationale",
    "tech_research",
    "market_result",
    "stakeholder_result",
    "domain_result",
    "synthesis",
)


def _validate_state(state: dict) -> None:
    """보고서 생성에 필요한 State 값과 중첩 필드를 검사한다."""
    missing = [
        key
        for key in _REQUIRED_STATE_KEYS
        if key not in state or state[key] is None
    ]

    if missing:
        raise ValueError(
            "보고서 생성에 필요한 State 값이 없습니다: "
            + ", ".join(missing)
        )

    nested_requirements = {
        "tech_research": ("sw", "hw"),
        "market_result": ("sw", "hw"),
        "stakeholder_result": ("sw", "hw"),
        "domain_result": ("sw", "hw"),
        "synthesis": ("agreements", "conflicts", "implications"),
    }
    missing_nested = [
        f"{parent}.{child}"
        for parent, children in nested_requirements.items()
        for child in children
        if not isinstance(state[parent], dict) or child not in state[parent]
    ]
    if missing_nested:
        raise ValueError(
            "보고서 생성에 필요한 State 값이 없습니다: "
            + ", ".join(missing_nested)
        )


def _format_prompt_value(value) -> str:
    """dict/list 입력을 LLM이 읽기 쉬운 JSON 문자열로 변환한다."""
    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
    )


def _strip_reference_section(report: str) -> str:
    """LLM이 실수로 생성한 REFERENCE 섹션을 제거한다."""
    match = re.search(
        r"(?im)^##\s*(?:7\.\s*)?REFERENCE\s*$",
        report,
    )

    if match:
        return report[:match.start()].rstrip()

    return report.strip()

# 누락된 목차가 없는지 확인
def _validate_report_structure(report: str) -> None:
    """설계서에서 요구한 목차가 정확히 포함됐는지 검사한다."""
    report_lines = {
        line.strip()
        for line in report.splitlines()
    }

    missing = [
        heading
        for heading in _REQUIRED_REPORT_HEADINGS
        if heading not in report_lines
    ]

    if missing:
        raise ValueError(
            "보고서 필수 목차가 누락됐습니다: "
            + ", ".join(missing)
        )

def _validate_citations(report: str, state: dict) -> None:
    """본문에서 인용한 출처 ID가 State에 존재하는지 검사한다."""
    error = _citation_validation_error(report, state)
    if error:
        raise ValueError(error)


def _citation_validation_error(report: str, state: dict) -> str | None:
    """인용 검증 실패 사유를 반환한다. 성공하면 None을 반환한다."""
    orphan_citations = find_orphan_citations(report, state)

    if orphan_citations:
        return "본문에 등록되지 않은 출처가 인용됐습니다: " + ", ".join(orphan_citations)

    if collect_sources(state) and not re.search(
        r"\[src_[A-Za-z0-9_]+\]",
        report,
    ):
        return "보고서 본문에 출처 인용이 없습니다"

    return None


def _citation_repair_prompt(
    prompt: str,
    draft: str,
    issue: str,
    state: dict,
) -> str:
    """인용 누락·오류가 발생했을 때 완전한 보고서 본문을 한 번만 재생성하도록 지시한다."""
    source_ids = [
        str(source["id"])
        for source in collect_sources(state)
        if source.get("id")
    ]
    allowed_ids = ", ".join(source_ids) or "(등록된 출처 없음)"
    return f"""{prompt}

[인용 보완 재생성 — 반드시 수행]
아래 초안은 인용 검증에 실패했습니다.
실패 사유: {issue}
사용 가능한 출처 ID: {allowed_ids}

초안의 사실관계와 필수 목차를 유지하되, 사실 주장 뒤에 위 목록의 ID만 `[src_xxx]` 형식으로
추가하여 보고서 본문 전체를 다시 작성하세요. 새로운 사실·출처·REFERENCE 섹션은 추가하지 마세요.
인용이 하나도 없는 응답은 허용되지 않습니다.

[인용 보완 전 초안]
{draft}
"""

def _source_identifier(source: dict) -> str:
    return str(
        source.get("id")
        or source.get("url")
        or source.get("title")
        or "unknown"
    )


def report_node(state: dict) -> dict:
    _validate_state(state)

    tech_research = state["tech_research"]
    synthesis = state["synthesis"]

    prompt = load_prompt("report").format(
        selection_rationale=state["selection_rationale"],
        sw_research=_format_prompt_value(tech_research["sw"]),
        hw_research=_format_prompt_value(tech_research["hw"]),
        trl_sw=tech_research["sw"].get("trl", "근거 확인 불가"),
        trl_hw=tech_research["hw"].get("trl", "근거 확인 불가"),
        market_sw=state["market_result"].get(
            "sw",
            "공개된 근거로 확인하기 어렵다",
        ),
        market_hw=state["market_result"].get(
            "hw",
            "공개된 근거로 확인하기 어렵다",
        ),
        stakeholder_sw=state["stakeholder_result"].get(
            "sw",
            "공개된 근거로 확인하기 어렵다",
        ),
        stakeholder_hw=state["stakeholder_result"].get(
            "hw",
            "공개된 근거로 확인하기 어렵다",
        ),
        domain=state.get(
            "domain",
            "Cloud/Data Center 환경의 Long-context LLM Serving",
        ),
        domain_sw=state["domain_result"].get(
            "sw",
            "공개된 근거로 확인하기 어렵다",
        ),
        domain_hw=state["domain_result"].get(
            "hw",
            "공개된 근거로 확인하기 어렵다",
        ),
        agreements=_format_prompt_value(
            synthesis.get("agreements", "확인 불가")
        ),
        conflicts=_format_prompt_value(
            synthesis.get("conflicts", "확인 불가")
        ),
        implications=_format_prompt_value(
            synthesis.get("implications", "확인 불가")
        ),
        evidence_status=(
            "충분"
            if state.get("evidence_sufficient", False)
            else "불충분"
        ),
        retry_count=state.get("retry_count", 0),
        warnings=_format_prompt_value(state.get("warnings", [])),
        available_sources=_format_prompt_value(collect_sources(state)),
    )

    response = get_llm(temperature=0.2).invoke(prompt)
    report_body = _strip_reference_section(str(response.content))

    # 인용이 누락되거나 등록되지 않은 인용이 있으면, 같은 입력으로 1회만 보완 생성한다.
    citation_issue = _citation_validation_error(report_body, state)
    if citation_issue:
        repair_prompt = _citation_repair_prompt(
            prompt,
            report_body,
            citation_issue,
            state,
        )
        repaired = get_llm(temperature=0.2).invoke(repair_prompt)
        report_body = _strip_reference_section(str(repaired.content))

    _validate_report_structure(report_body)
    _validate_citations(report_body, state)

    reference_text = render_references(
        state,
        report_md=report_body,
    )

    final_report = (
        f"{report_body}\n\n"
        f"## 7. REFERENCE\n\n"
        f"{reference_text}\n"
    )

    warnings = list(state.get("warnings", []))
    source_audit = audit_sources(state)

    incomplete_sources = source_audit["incomplete"]
    if incomplete_sources:
        identifiers = ", ".join(
            _source_identifier(source)
            for source in incomplete_sources
        )
        warnings.append(
            f"[report] 서지정보가 불완전한 출처: {identifiers}"
        )

    dropped_sources = source_audit["dropped"]
    if dropped_sources:
        warnings.append(
            "[report] 출처가 아닌 값이 sources에서 제외됨: "
            + ", ".join(map(str, dropped_sources))
        )

    return {
        "final_report": final_report,
        "warnings": warnings,
    }
