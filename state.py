"""LangGraph 전체 State 스키마 정의.

병렬로 실행되는 시장성/이해관계자/도메인 평가 에이전트가 서로 다른 키에 쓰도록 설계하여
fan-out/fan-in 시 State 충돌이 없도록 함 (설계 문서 D. 그래프 설계(안) 참고).
"""
from typing import TypedDict, List, Dict, Any, Optional


class TechCandidate(TypedDict):
    id: str
    camp: str          # "SW" | "HW"
    title: str
    year: str
    summary: str
    url: str
    file: str           # data/ 폴더에 두어야 하는 로컬 PDF 파일명


class TechReport(TypedDict, total=False):
    overview: str
    approach: str
    limitations: str
    trl: str  # 기술성숙도(TRL) 관점 — 논문/구현체/실제검증/생태계통합/상용화 근거 기반 추정
    sources: List[str]


class PerspectiveResult(TypedDict, total=False):
    sw: str
    hw: str
    summary: str
    sources: List[str]


class GraphState(TypedDict, total=False):
    # 입력 / 후보 풀
    sw_pool: List[TechCandidate]
    hw_pool: List[TechCandidate]
    domain_focus: str  # 기본값: "Cloud/Data Center 환경의 Long-context LLM Serving"

    # 1. 기술 선정
    selected_sw: TechCandidate
    selected_hw: TechCandidate
    selection_rationale: str

    # 2. 기술 조사 (RAG)
    tech_research: Dict[str, TechReport]  # {"sw": {...}, "hw": {...}}

    # 3. 병렬 평가 (fan-out)
    market_result: PerspectiveResult       # RAG
    stakeholder_result: PerspectiveResult  # No RAG
    domain_result: PerspectiveResult       # RAG

    # 4. 종합 / 보고서
    synthesis: Dict[str, Any]
    final_report: str

    # 진행 중 경고/에러 로그 (예: PDF 파일 누락 등)
    warnings: List[str]
