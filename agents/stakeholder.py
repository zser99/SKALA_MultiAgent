"""기술조사 결과를 이해관계자 관점으로 분석하는 단일 노드.

그래프 연결은 graph.py에서 담당하며, 이 모듈은 stakeholder_result만 반환한다.
초안의 평가 기준과 프롬프트를 사용하고 별도의 벡터 검색은 수행하지 않는다.
보완 검색은 tools.tavily_web_search 모듈에 위임한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.utils import load_prompt
from llm import get_llm
from state import GraphState, PerspectiveResult


# 실행 위치와 관계없이 프로젝트의 평가 기준 파일을 읽는다.
_RUBRIC_PATH = Path(__file__).resolve().parent.parent / "prompts" / "stakeholder_rubric.json"
with _RUBRIC_PATH.open(encoding="utf-8") as _rubric_file:
    STAKEHOLDER_RUBRIC: dict[str, dict[str, Any]] = json.load(_rubric_file)



EvidenceLevel = Literal["sufficient", "partial", "insufficient"]


### LLM이 분석 결과를 정해진 형식으로 반환하도록 만드는 내부 출력 양식 ###
# 기술 하나(SW 또는 HW)를 특정 평가 기준으로 분석한 결과
class TechnologyAssessment(BaseModel):
    observed_items: list[str] = Field(default_factory=list, description="직접 근거가 확인된 관찰 항목. 가능하면 Rubric 항목명을 사용하되 설명형 표현도 허용한다.")
    evidence_level: EvidenceLevel = "insufficient"
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    direct_evidence: list[str] = Field(default_factory=list)
    labeled_inferences: list[str] = Field(default_factory=list)
# 이해관계자의 평가 기준 하나에 대한 SW·HW 분석 및 비교 결과
class CriterionAssessment(BaseModel):
    stakeholder: str
    criterion: str
    sw: TechnologyAssessment
    hw: TechnologyAssessment
    tradeoff: str | None = None
    source_urls: list[str] = Field(default_factory=list)
# 5개 이해관계자 × 3개 평가 기준의 분석을 모은 LLM 내부 출력 형식
# 이 결과를 정리하여 최종적으로 state의 stakeholder_result에 반환한다.
class StakeholderBatchAssessment(BaseModel):
    criteria: list[CriterionAssessment]


# 관점 하나의 세 평가 기준을 종합한 최종 판단 형식
class StakeholderConclusion(BaseModel):
    stakeholder: str
    conclusion: str = Field(min_length=1)


# 다섯 관점의 종합 판단을 한 번에 받는 내부 출력 형식
class StakeholderConclusions(BaseModel):
    conclusions: list[StakeholderConclusion]


CONCLUSION_HEADING = "[이해관계자별 최종 종합 결과]"


# 최종 평가의 장단점과 근거를 종합하여 관점별 판단을 한 문단씩 생성한다.
def _generate_conclusions(
    batch: StakeholderBatchAssessment,
    sw_title: str,
    hw_title: str,
    tech_research: dict[str, Any],
    web_result: dict[str, Any],
) -> str:
    model = get_llm().with_structured_output(StakeholderConclusions)
    payload = {
        "sw_title": sw_title,
        "hw_title": hw_title,
        "rubric": STAKEHOLDER_RUBRIC,
        "assessment": batch.model_dump(),
        "tech_research": tech_research,
        "web_evidence": {"results": web_result.get("results", [])},
    }
    result = model.invoke([
        SystemMessage(content=load_prompt("stakeholder_conclusion").strip()),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])
    conclusions = {item.stakeholder: item.conclusion.strip() for item in result.conclusions}
    if (
        len(result.conclusions) != len(STAKEHOLDER_RUBRIC)
        or set(conclusions) != set(STAKEHOLDER_RUBRIC)
        or not all(conclusions.values())
    ):
        raise ValueError("종합 판단은 정의된 다섯 관점에 대해 각각 한 개씩 필요합니다.")

    lines = [CONCLUSION_HEADING]
    for stakeholder in STAKEHOLDER_RUBRIC:
        conclusion = conclusions[stakeholder]
        lines.extend(["", f"[{stakeholder}]", f"최종 종합 결과: {conclusion}"])
    return "\n".join(lines)


# 시스템 프롬프트 가져오기
STAKEHOLDER_SYSTEM_PROMPT = load_prompt("stakeholder_system").strip()


# 자료를 프롬프트용 JSON 문자열로 변환 & 최대 글자 수를 넘으면 자르기
def _json_text(value: Any, max_chars: int = 80_000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text

# 이해관계자별 관점과 평가 기준을 시스템 프롬프트에 넣을 문자열로 구성
def _stakeholder_context_text() -> str:
    blocks: list[str] = []
    for stakeholder, config in STAKEHOLDER_RUBRIC.items():
        blocks.append(f"[{stakeholder}]\n관점: {config['perspective']}")
        for c in config["criteria"]:
            blocks.append(
                f"- 평가 기준: {c['criterion']}\n"
                f"  관찰 항목: {', '.join(c['items'])}\n"
                f"  참고 프레임: {', '.join(c['references'])}"
            )
    return "\n".join(blocks)

# 이해관계자와 평가 기준 이름으로 해당 기준의 상세 내용을 조회할 사전 생성
def _criterion_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for stakeholder, config in STAKEHOLDER_RUBRIC.items():
        for spec in config["criteria"]:
            lookup[(stakeholder, spec["criterion"])] = spec
    return lookup
# 만든 사전 저장
CRITERION_LOOKUP = _criterion_lookup()

# 관점 생성 근거 항목별 점수화
def _normalized_level(observed_count: int) -> EvidenceLevel:
    if observed_count >= 2:
        return "sufficient"
    if observed_count == 1:
        return "partial"
    return "insufficient"

# 분석 결과 검증 및 보정
def _sanitize_assessment(batch: StakeholderBatchAssessment) -> StakeholderBatchAssessment:
    """관찰 항목과 직접 근거를 확인하여 근거 수준을 보정한다."""
    expected = set(CRITERION_LOOKUP.keys())
    seen: set[tuple[str, str]] = set()
    cleaned: list[CriterionAssessment] = []

    for item in batch.criteria:
        key = (item.stakeholder, item.criterion)
        if key not in expected or key in seen:
            continue
        seen.add(key)

        for view in (item.sw, item.hw):
            # 표현 차이로 근거를 삭제하지 않고, 정확히 매핑된 항목만 개수 판정에 사용한다.
            allowed_items = CRITERION_LOOKUP[key]["items"]
            view.direct_evidence = list(dict.fromkeys(
                text.strip() for text in view.direct_evidence if text.strip()
            ))
            view.observed_items = list(dict.fromkeys(
                text.strip() for text in view.observed_items if text.strip()
            )) if view.direct_evidence else []
            mapped_count = sum(text in allowed_items for text in view.observed_items)
            if not view.direct_evidence:
                view.evidence_level = "insufficient"
            elif mapped_count:
                view.evidence_level = _normalized_level(mapped_count)
            else:
                # 문장 존재만으로 근거 충분을 인정하지 않고 기존 판단을 보수적으로 유지한다.
                if view.evidence_level == "sufficient":
                    view.evidence_level = "partial"
            view.labeled_inferences = [
                text if text.startswith("추론:") else f"추론: {text}"
                for text in view.labeled_inferences if text.strip()
            ]

        if not item.sw.direct_evidence:
            item.sw.pros = []
            item.sw.cons = []
            item.sw.direct_evidence = []
            item.sw.labeled_inferences = []

        if not item.hw.direct_evidence:
            item.hw.pros = []
            item.hw.cons = []
            item.hw.direct_evidence = []
            item.hw.labeled_inferences = []

        if item.sw.evidence_level == "insufficient" and item.hw.evidence_level == "insufficient":
            item.tradeoff = "Insufficient Evidence"
        elif item.sw.evidence_level == "insufficient" or item.hw.evidence_level == "insufficient":
            item.tradeoff = "한쪽 기술의 직접 근거가 부족하여 해당 기준의 직접 비교는 제한적임."

        cleaned.append(item)

    # 누락된 평가 기준은 근거 부족으로 명시한다.
    for stakeholder, criterion in expected - seen:
        cleaned.append(
            CriterionAssessment(
                stakeholder=stakeholder,
                criterion=criterion,
                sw=TechnologyAssessment(),
                hw=TechnologyAssessment(),
                tradeoff="Insufficient Evidence",
            )
        )

    # 초안의 이해관계자와 평가 기준 순서를 유지한다.
    order = {
        key: idx
        for idx, key in enumerate(
            (stakeholder, spec["criterion"])
            for stakeholder, cfg in STAKEHOLDER_RUBRIC.items()
            for spec in cfg["criteria"]
        )
    }
    cleaned.sort(key=lambda x: order[(x.stakeholder, x.criterion)])
    return StakeholderBatchAssessment(criteria=cleaned)

# 웹 보완 검색이 필요한 대상 만들기
def _missing_specs(batch: StakeholderBatchAssessment) -> list[dict[str, Any]]:
    """관점별로 모든 기준에 근거가 없는 기술의 검색 대상을 모은다."""
    by_stakeholder: dict[str, list[CriterionAssessment]] = {}
    for item in batch.criteria:
        by_stakeholder.setdefault(item.stakeholder, []).append(item)

    missing: list[dict[str, Any]] = []
    for stakeholder, assessments in by_stakeholder.items():
        # 항목명 매핑 여부와 관계없이 세 기준 모두 근거 내용이 없는 기술만 선택한다.
        missing_for = [
            side
            for side in ("sw", "hw")
            if all(
                not getattr(item, side).direct_evidence
                for item in assessments
            )
        ]
        if not missing_for:
            continue

        # 선택된 기술에 대해 해당 관점의 세 기준을 검색 대상으로 전달한다.
        for item in assessments:
            spec = CRITERION_LOOKUP[(stakeholder, item.criterion)]
            missing.append(
                {
                    "stakeholder": stakeholder,
                    "criterion": item.criterion,
                    "items": spec["items"],
                    "missing_for": missing_for.copy(),
                }
            )
    return missing


def _merge_web_update(
    base: StakeholderBatchAssessment,
    update: StakeholderBatchAssessment,
) -> StakeholderBatchAssessment:
    base_map = {(x.stakeholder, x.criterion): x for x in base.criteria}
    update = _sanitize_assessment(update)

    for new_item in update.criteria:
        key = (new_item.stakeholder, new_item.criterion)
        if key not in base_map:
            continue
        old = base_map[key]

        # 기존에 근거가 없던 기술 측만 직접 근거를 확보했을 때 보완한다.
        if not old.sw.direct_evidence and new_item.sw.direct_evidence:
            old.sw = new_item.sw
        if not old.hw.direct_evidence and new_item.hw.direct_evidence:
            old.hw = new_item.hw

        valid_urls = [u for u in new_item.source_urls if u]
        old.source_urls = list(dict.fromkeys(old.source_urls + valid_urls))

        if old.sw.evidence_level != "insufficient" and old.hw.evidence_level != "insufficient":
            old.tradeoff = new_item.tradeoff or old.tradeoff
        elif old.sw.evidence_level == "insufficient" and old.hw.evidence_level == "insufficient":
            old.tradeoff = "Insufficient Evidence"
        else:
            old.tradeoff = "한쪽 기술의 직접 근거가 부족하여 해당 기준의 직접 비교는 제한적임."

    return _sanitize_assessment(
        StakeholderBatchAssessment(criteria=list(base_map.values()))
    )

# 기존 분석과 웹 보완 분석의 병합 함수
def _bullet_lines(values: list[str], empty_text: str) -> list[str]:
    if not values:
        return [f"  - {empty_text}"]
    return [f"  - {v}" for v in values]

# SW 또는 HW의 분석 결과를 사람이 읽을 수 있는 보고서 문자열로 정리
def _render_technology(batch: StakeholderBatchAssessment, side: Literal["sw", "hw"]) -> str:
    lines: list[str] = []
    by_stakeholder: dict[str, list[CriterionAssessment]] = {}
    for item in batch.criteria:
        by_stakeholder.setdefault(item.stakeholder, []).append(item)

    for stakeholder in STAKEHOLDER_RUBRIC.keys():
        lines.append(f"[{stakeholder}]")
        for idx, item in enumerate(by_stakeholder.get(stakeholder, []), start=1):
            view: TechnologyAssessment = getattr(item, side)
            lines.append(f"{idx}. {item.criterion}")
            lines.append(f"- Evidence Level: {view.evidence_level.capitalize()}")

            if not view.direct_evidence:
                lines.append("- 평가: Insufficient Evidence")
                lines.append("- 근거: 제공된 자료에서 직접 근거를 확보하지 못함.")
                continue

            if view.evidence_level == "insufficient":
                lines.append("- 안내: 근거 내용은 있으나 관찰 항목의 연결 또는 근거 수준 확인이 필요함.")
            lines.append("- 확인된 항목:")
            lines.extend(_bullet_lines(view.observed_items, "직접 확인된 세부 항목 없음"))
            lines.append("- Pros:")
            lines.extend(_bullet_lines(view.pros, "직접 근거 기반 Pros 없음"))
            lines.append("- Cons:")
            lines.extend(_bullet_lines(view.cons, "직접 근거 기반 Cons 없음"))
            lines.append("- Direct Evidence:")
            lines.extend(_bullet_lines(view.direct_evidence, "직접 인용 가능한 근거 없음"))
            if view.labeled_inferences:
                lines.append("- Inference:")
                lines.extend(_bullet_lines(view.labeled_inferences, ""))
        lines.append("")

    return "\n".join(lines).strip()

# 이해관계자별 SW·HW 트레이드오프와 근거 부족 현황을 요약하는 함수
def _render_summary(batch: StakeholderBatchAssessment, sw_title: str, hw_title: str) -> str:
    lines = [
        f"{sw_title}(SW)와 {hw_title}(HW)를 Data Center / Cloud 산업의 이해관계자 관점에서 비교하였다.",
        "절대적 우위나 점수는 산출하지 않았으며, 직접 근거가 확보된 기준만 Pros/Cons/Trade-off 분석에 사용하였다.",
        "",
    ]

    current = None
    for item in batch.criteria:
        if item.stakeholder != current:
            current = item.stakeholder
            lines.append(f"[{current}]")
        tradeoff = item.tradeoff or "직접 비교 가능한 Trade-off 근거가 부족함."
        lines.append(f"- {item.criterion}: {tradeoff}")

    insufficient = sum(
        1
        for item in batch.criteria
        if item.sw.evidence_level == "insufficient" or item.hw.evidence_level == "insufficient"
    )
    if insufficient:
        lines.extend(
            [
                "",
                f"※ 총 {insufficient}개 평가 기준에서 SW 또는 HW 한쪽 이상의 직접 근거가 부족해 비교가 제한되었다.",
            ]
        )

    return "\n".join(lines).strip()

# 기술조사 및 이해관계자 분석의 출처를 수집하고 중복을 제거하는 함수
def _collect_sources(state: GraphState, batch: StakeholderBatchAssessment) -> list[str]:
    sources: list[str] = []
    tech_research = state.get("tech_research", {}) or {}
    for side in ("sw", "hw"):
        report = tech_research.get(side, {}) or {}
        for url in report.get("sources", []) or []:
            if url and url not in sources:
                sources.append(url)

    for item in batch.criteria:
        for url in item.source_urls:
            if url and url not in sources:
                sources.append(url)
    return sources

# 최종 노드(에이전트)
# 이거를 graph.py에 연결하면 된다.
def stakeholder_node(state: GraphState) -> dict[str, PerspectiveResult]:
    """기술조사 근거를 관점별로 평가하고 이해관계자 결과만 반환한다."""
    started_at = perf_counter()
    print("[stakeholder] 분석 시작", flush=True)
    structured_model = get_llm().with_structured_output(StakeholderBatchAssessment)
    system_prompt = STAKEHOLDER_SYSTEM_PROMPT.format(
        stakeholder_context=_stakeholder_context_text()
    )
    selected_sw = state.get("selected_sw", {}) or {}
    selected_hw = state.get("selected_hw", {}) or {}
    sw_title = selected_sw.get("title", "SW 후보 기술")
    hw_title = selected_hw.get("title", "HW 후보 기술")
    tech_research = state.get("tech_research", {}) or {}

    # 기술조사 에이전트가 전달한 근거로 전체 기준을 평가한다.
    rag_prompt = f"""
비교 기술:
- SW: {sw_title}
- HW: {hw_title}

upstream RAG 기술 조사 결과(tech_research):
{_json_text(tech_research)}

위 자료를 최우선 근거로 사용하여 아래 5개 이해관계자 × 3개 평가 기준을 모두 분석하라.
각 기술별 observed_items 수에 따라 evidence_level을 판정하라.
근거가 없는 세부 항목은 채우지 마라.

반드시 STAKEHOLDER_RUBRIC에 정의된 정확한 stakeholder 이름과 criterion 이름을 사용하라.
""".strip()

    print("[stakeholder] 기술조사 결과 기반 관점별 LLM 분석 시작", flush=True)
    step_started_at = perf_counter()
    try:
        initial = structured_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=rag_prompt),
            ]
        )
    except Exception as exc:
        print(f"[stakeholder] 기술조사 결과 기반 관점별 LLM 분석 실패 ({type(exc).__name__})", flush=True)
        raise
    print(f"[stakeholder] 기술조사 결과 기반 관점별 LLM 분석 완료 ({perf_counter() - step_started_at:.1f}초)", flush=True)
    assessment = _sanitize_assessment(initial)
    insufficient_before = {
        side: sum(getattr(item, side).evidence_level == "insufficient" for item in assessment.criteria)
        for side in ("sw", "hw")
    }
    print(
        f"[stakeholder] 결과 검증·보정 완료 — 근거 부족: "
        f"SW {insufficient_before['sw']}/{len(assessment.criteria)}, "
        f"HW {insufficient_before['hw']}/{len(assessment.criteria)}개 기준",
        flush=True,
    )

    # 관점별로 근거가 전혀 없는 기술의 기준을 모아 한 번만 보완 검색한다.
    web_result: dict[str, Any] = {"results": []}
    missing = _missing_specs(assessment)
    if missing:
        targets = dict.fromkeys(
            (spec["stakeholder"], side)
            for spec in missing for side in spec["missing_for"]
        )
        for stakeholder, side in targets:
            print(f"[stakeholder] 웹 보완 대상: {stakeholder} — {side.upper()}", flush=True)
        print("[stakeholder] 통합 웹 검색 시작 (최대 1회)", flush=True)
        step_started_at = perf_counter()
        try:
            # 검색 도구를 사용할 수 없더라도 기술조사 기반 평가는 유지한다.
            from tools.tavily_web_search import search_missing_evidence

            web_result = search_missing_evidence(
                sw_title,
                hw_title,
                missing,
            )
        except Exception as exc:
            # 예외 원문에는 인증 정보가 포함될 수 있으므로 오류 종류만 출력한다.
            print(
                f"[stakeholder] 웹 검색 실패 ({type(exc).__name__}) — 기존 분석 유지",
                flush=True,
            )
            web_result = {
                "query": "",
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            print(
                f"[stakeholder] 웹 검색 완료 — 결과 {len(web_result.get('results') or [])}건 "
                f"({perf_counter() - step_started_at:.1f}초)",
                flush=True,
            )
            if not web_result.get("results"):
                print("[stakeholder] 검색 결과 없음 — 웹 보완 분석 생략, 기존 분석 유지", flush=True)

        if web_result.get("results"):
            web_prompt = f"""
비교 기술:
- SW: {sw_title}
- HW: {hw_title}

RAG에서 직접 근거가 부족했던 기준:
{_json_text(missing, max_chars=20_000)}

보완 웹 검색 결과(단 1회):
{_json_text(web_result, max_chars=50_000)}

규칙:
- 위 '근거 부족 기준'만 재평가한다.
- 기존에 충분/부분 근거가 있던 기술 측은 새로 작성하지 않아도 된다.
- 검색 결과가 해당 기술과 해당 평가항목을 직접 뒷받침할 때만 observed_items와 direct_evidence에 반영한다.
- 일반적인 CXL/GPU/CUDA/NVMe 설명은 특정 기술의 직접 근거로 사용하지 않는다.
- 웹 검색에서도 직접 근거가 없으면 evidence_level='insufficient'를 유지한다.
- 사용한 웹 근거 URL만 source_urls에 기록한다.
- 절대적 우위, 점수, 순위를 만들지 않는다.
""".strip()

            print("[stakeholder] 웹 보완 LLM 분석 시작", flush=True)
            step_started_at = perf_counter()
            try:
                web_update = structured_model.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=web_prompt),
                    ]
                )
            except Exception as exc:
                print(f"[stakeholder] 웹 보완 LLM 분석 실패 ({type(exc).__name__})", flush=True)
                raise
            print(f"[stakeholder] 웹 보완 LLM 분석 완료 ({perf_counter() - step_started_at:.1f}초)", flush=True)
            print("[stakeholder] 결과 병합 시작", flush=True)
            assessment = _merge_web_update(assessment, web_update)
            recovered = {
                side: insufficient_before[side] - sum(
                    getattr(item, side).evidence_level == "insufficient"
                    for item in assessment.criteria
                )
                for side in ("sw", "hw")
            }
            print(
                f"[stakeholder] 결과 병합 완료 — 근거 확보: SW {recovered['sw']}개, HW {recovered['hw']}개 기준",
                flush=True,
            )
    else:
        print("[stakeholder] 보완 대상 없음 — 웹 검색 생략", flush=True)

    print("[stakeholder] 관점별 최종 종합 판단 생성 시작", flush=True)
    step_started_at = perf_counter()
    try:
        conclusions = _generate_conclusions(assessment, sw_title, hw_title, tech_research, web_result)
    except Exception as exc:
        print(f"[stakeholder] 관점별 최종 종합 판단 생성 실패 ({type(exc).__name__})", flush=True)
        raise
    print(f"[stakeholder] 관점별 최종 종합 판단 5개 생성 완료 ({perf_counter() - step_started_at:.1f}초)", flush=True)

    # 내부의 상세 평가를 공통 상태의 결과 형식으로 변환한다.
    print("[stakeholder] stakeholder_result 생성 시작", flush=True)
    stakeholder_result: PerspectiveResult = {
        # 후속 에이전트는 요약 필드를 읽지 않으므로 기존 전달 필드에도 종합 판단을 붙인다.
        "sw": _render_technology(assessment, "sw") + "\n\n" + conclusions,
        "hw": _render_technology(assessment, "hw"),
        "summary": _render_summary(assessment, sw_title, hw_title) + "\n\n" + conclusions,
        "sources": _collect_sources(state, assessment),
    }

    # 병렬 노드와 충돌하지 않도록 담당하는 상태 키만 갱신한다.
    print(f"[stakeholder] stakeholder_result 생성 완료 (총 {perf_counter() - started_at:.1f}초)", flush=True)
    return {"stakeholder_result": stakeholder_result}
