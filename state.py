"""LangGraph 전체 State 스키마 정의.

병렬로 실행되는 시장성/이해관계자/도메인 평가 에이전트가 서로 다른 키에 쓰도록 설계하여
fan-out/fan-in 시 State 충돌이 없도록 함 (설계 문서 D. 그래프 설계(안) 참고).
"""
from typing import Annotated, TypedDict, List, Dict, Any, Literal, Union


def merge_worker_results(current: dict, updates: dict) -> dict:
    return {**current, **updates}


class WorkItem(TypedDict):
    task_id: str
    worker: Literal["market", "stakeholder", "domain"]
    instruction: str
    reason: str
    attempt: int


class WorkerResult(TypedDict):
    worker: str
    status: Literal["completed", "failed"]
    attempt: int
    payload: Dict[str, Any]
    error: str | None
    evidence_path: str


class EvalVerdict(TypedDict):
    passed: bool
    checks: Dict[str, bool]
    reasons: List[str]
    rework_targets: List[str]


class TechCandidate(TypedDict):
    id: str
    camp: str          
    title: str
    year: str
    summary: str
    url: str
    file: str           



class SourceBase(TypedDict):
    id: str
    type: Literal["paper", "patent", "web"]
    title: str


class SourceRecord(SourceBase, total=False):
    # paper
    authors: str
    year: str
    venue: str
    pages: str

    # patent
    applicant: str
    date: str
    number: str

    # web
    org: str
    author: str
    site: str

    # common
    url: str


# 기존 Agent가 아직 문자열 출처를 반환하므로 전환 기간 동안 둘 다 허용
SourceItem = Union[str, SourceRecord]

class TechReport(TypedDict, total=False):
    overview: str
    approach: str
    limitations: str
    trl: str
    sources: List[SourceItem]


class PerspectiveResult(TypedDict, total=False):
    sw: str
    hw: str
    summary: str
    sources: List[SourceItem]
    limitations: Dict[str, List[str]]
    basis: Dict[str, str]
    domain_analysis: Dict[str, Any]
    domain_sources: List[Dict[str, Any]]


class GraphState(TypedDict, total=False):
    evaluation_request: str
    required_perspectives: List[str]
    run_id: str
    plan: List[WorkItem]
    worker_results: Annotated[Dict[str, WorkerResult], merge_worker_results]
    round_count: int
    max_rounds: int
    step_count: int
    max_steps: int
    last_error: str | None
    termination_reason: str
    eval_result: EvalVerdict
    quality_feedback: List[str]
    generation_error: str | None
    task: WorkItem
    # 입력 / 후보 풀
    sw_pool: List[TechCandidate]
    hw_pool: List[TechCandidate]
    domain: str

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
    domain_source: List[str]

    # 4. 근거 충분성 검증 (Evidence Check -> 최대 1회 재검색)
    evidence_sufficient: bool
    retry_count: int

    # 5. 종합 / 보고서
    synthesis: Dict[str, Any]
    final_report: str

    # 진행 중 경고/에러 로그 (예: PDF 파일 누락 등)
    warnings: List[str]
