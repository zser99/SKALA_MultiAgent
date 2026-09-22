from typing import TypedDict, List, Dict, Any, Optional


class TechCandidate(TypedDict):
    id: str
    camp: str          
    title: str
    year: str
    summary: str
    url: str
    file: str           


class TechReport(TypedDict, total=False):
    overview: str
    approach: str
    limitations: str
    trl: str  
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
