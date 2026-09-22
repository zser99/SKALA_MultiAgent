"""REFERENCE 생성 및 인용 검증 (보고서 생성 에이전트 전용).

참고문헌은 LLM에게 맡기지 않고 코드로 만든다. LLM이 쓰면 없는 논문을 지어내고,
과제 요건인 "보고서 작성에 실제로 활용한 자료 목록만 기재"가 깨지기 때문이다.

표기 형식 (과제 지정):
    특허 : 출원인(YYYY-MM). 특허명, 특허번호/공개번호, URL
    논문 : 저자(YYYY). 논문제목. 학술지/학회명, 권(호), 페이지.
    기타 : 기관명 또는 작성자(YYYY-MM-DD). 제목. 사이트명, URL

State의 sources는 현재 두 가지 형태가 섞여 있다.
    - dict : 아래 Source 스펙을 따르는 구조화된 출처 (권장)
    - str  : URL 또는 레지스트리 키 (현행)
둘 다 처리하되, dict 쪽이 정확한 표기를 만들 수 있으므로 각 에이전트 담당자에게
tests/mock_state.py 의 SOURCE_REGISTRY 형태를 요청할 것.
"""
from typing import Any, Iterable, Optional

# sources 자리에 들어오지만 출처가 아닌 값들 (market 노드의 basis 라벨 등)
_NON_SOURCE_PREFIXES = ("sw_basis=", "hw_basis=", "basis=")

_PERSPECTIVE_KEYS = ("market_result", "stakeholder_result", "domain_result")
_PLACEHOLDER_MARKERS = (
    "확인 필요",
    "미상",
    "example.com",
)


def _is_incomplete_source(source: dict) -> bool:
    """출처의 필수 서지정보 누락 또는 임시 값을 검사한다."""
    if source.get("_incomplete"):
        return True

    source_type = source.get("type")
    if source_type not in {"paper", "patent", "web"}:
        return True

    if any(
        marker in str(value)
        for value in source.values()
        for marker in _PLACEHOLDER_MARKERS
    ):
        return True

    common_fields = ("id", "title")
    if any(not source.get(field) for field in common_fields):
        return True

    if source_type == "paper":
        required_fields = ("authors", "year", "venue")
        return any(not source.get(field) for field in required_fields)

    if source_type == "patent":
        required_fields = ("applicant", "date", "number", "url")
        return any(not source.get(field) for field in required_fields)

    # web
    required_fields = ("date", "site", "url")
    if any(not source.get(field) for field in required_fields):
        return True

    return not (source.get("org") or source.get("author"))

def _is_real_source(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("id") or value.get("url") or value.get("title"))
    if isinstance(value, str):
        text = value.strip()
        if not text or text.startswith(_NON_SOURCE_PREFIXES):
            return False
        return True
    return False


def _normalize(value: Any, registry: Optional[dict] = None) -> Optional[dict]:
    """sources 항목 하나를 Source dict로 정규화한다."""
    if isinstance(value, dict):
        return value

    text = str(value).strip()
    if registry:
        if text in registry:
            return registry[text]
        # URL로 들어온 출처도 레지스트리에 등록돼 있으면 서지 정보를 붙여준다.
        for entry in registry.values():
            if entry.get("url") and entry["url"].rstrip("/") == text.rstrip("/"):
                return entry

    if text.startswith("http"):
        kind = "paper" if "arxiv.org" in text else "web"
        return {"id": text, "type": kind, "url": text, "title": "", "_incomplete": True}

    # 레지스트리에 없는 식별자 — 형식을 만들 수 없으므로 표시해 둔다.
    return {"id": text, "type": "unknown", "title": "", "_incomplete": True}


def collect_sources(state: dict, registry: Optional[dict] = None) -> list[dict]:
    """State 전체를 훑어 실제 사용된 출처를 모으고 중복을 제거한다."""
    raw: list[Any] = []

    tech_research = state.get("tech_research") or {}
    for side in ("sw", "hw"):
        raw.extend((tech_research.get(side) or {}).get("sources") or [])

    for key in _PERSPECTIVE_KEYS:
        raw.extend((state.get(key) or {}).get("sources") or [])

    # 선정된 기술의 원문은 항상 인용 대상이다.
    for key in ("selected_sw", "selected_hw"):
        cand = state.get(key)
        if cand and cand.get("url"):
            raw.append(
                {
                    "id": cand.get("id") or cand["url"],
                    "type": "paper",
                    "authors": cand.get("authors", ""),
                    "year": (cand.get("year") or "")[:4],
                    "title": cand.get("title", ""),
                    "venue": "arXiv preprint",
                    "url": cand["url"],
                }
            )

    # 같은 문헌이 URL로도 들어오고 구조화된 dict로도 들어온다(기술 조사의 sources vs
    # selected_*의 원문). URL과 id 양쪽으로 묶고, 서지 정보가 더 채워진 쪽을 남긴다.
    merged: dict[str, dict] = {}
    alias: dict[str, str] = {}

    def _completeness(src: dict) -> int:
        score = sum(1 for f in ("title", "authors", "year", "venue", "org", "date", "number") if src.get(f))
        return score - (5 if src.get("_incomplete") else 0)

    for item in raw:
        if not _is_real_source(item):
            continue
        src = _normalize(item, registry)
        if src is None:
            continue

        keys = [str(k) for k in (src.get("url"), src.get("id")) if k]
        canonical = next((alias[k] for k in keys if k in alias), keys[0] if keys else str(src.get("title")))
        for k in keys:
            alias[k] = canonical

        current = merged.get(canonical)
        if current is None or _completeness(src) > _completeness(current):
            merged[canonical] = {**(current or {}), **src}
        else:
            merged[canonical] = {**src, **current}

    return list(merged.values())


def format_reference(src: dict) -> str:
    """Source dict 하나를 과제 지정 표기 형식의 한 줄로 만든다."""
    kind = src.get("type", "unknown")

    if kind == "paper":
        authors = src.get("authors") or "(저자 미상)"
        year = src.get("year") or "n.d."
        title = src.get("title") or "(제목 미상)"
        venue = src.get("venue") or ""
        pages = src.get("pages") or ""
        tail = ", ".join(part for part in (venue, pages) if part)
        line = f"{authors}({year}). {title}."
        return f"{line} {tail}." if tail else line

    if kind == "patent":
        applicant = src.get("applicant") or "(출원인 미상)"
        date = src.get("date") or "n.d."
        title = src.get("title") or "(특허명 미상)"
        number = src.get("number") or "(번호 미상)"
        url = src.get("url") or ""
        return f"{applicant}({date}). {title}, {number}, {url}".rstrip(", ")

    if kind == "web":
        org = src.get("org") or src.get("author") or "(작성자 미상)"
        date = src.get("date") or "n.d."
        title = src.get("title") or "(제목 미상)"
        site = src.get("site") or ""
        url = src.get("url") or ""
        tail = ", ".join(part for part in (site, url) if part)
        return f"{org}({date}). {title}. {tail}".rstrip(", ")

    return f"(형식 미확인) {src.get('id') or src.get('url') or src.get('title')}"


def render_references(
    state: dict,
    registry: Optional[dict] = None,
    report_md: Optional[str] = None,
) -> str:
    """실제로 본문에서 인용된 출처만 REFERENCE로 만든다."""
    sources = collect_sources(state, registry)

    if report_md is not None:
        sources = [
            source
            for source in sources
            if (
                source.get("id")
                and f"[{source['id']}]" in report_md
            )
            or (
                source.get("url")
                and source["url"] in report_md
            )
        ]

        if not sources:
            return "(활용한 자료 없음)"

    groups = {
        "paper": [],
        "patent": [],
        "web": [],
        "unknown": [],
    }

    for source in sources:
        source_type = source.get("type", "unknown")
        groups.get(
            source_type,
            groups["unknown"],
        ).append(source)

    labels = {
        "paper": "논문",
        "patent": "특허",
        "web": "기타(웹)",
        "unknown": "확인 필요",
    }

    blocks = []

    for source_type in (
        "paper",
        "patent",
        "web",
        "unknown",
    ):
        items = groups[source_type]

        if not items:
            continue

        lines = [
            f"- {format_reference(source)}"
            for source in items
        ]

        blocks.append(
            f"**{labels[source_type]}**\n"
            + "\n".join(lines)
        )

    return "\n\n".join(blocks)


def audit_sources(state: dict, registry: Optional[dict] = None) -> dict:
    """제출 전 점검용. 표기를 완성할 수 없는 출처를 잡아낸다.

    Returns:
        {"total": int, "incomplete": [...], "dropped": [...]}
        incomplete — 유형/서지 정보가 부족해 형식을 못 맞추는 출처
        dropped    — sources 자리에 있었지만 출처가 아니라서 제외한 값
    """
    dropped: list[Any] = []

    tech_research = state.get("tech_research") or {}
    candidates: list[Any] = []
    for side in ("sw", "hw"):
        candidates.extend((tech_research.get(side) or {}).get("sources") or [])
    for key in _PERSPECTIVE_KEYS:
        candidates.extend((state.get(key) or {}).get("sources") or [])

    for item in candidates:
        if not _is_real_source(item):
            dropped.append(item)

    sources = collect_sources(state, registry)
    incomplete = [
        source
        for source in sources
        if _is_incomplete_source(source)
    ]
    return {"total": len(sources), "incomplete": incomplete, "dropped": dropped}


def find_orphan_citations(report_md: str, state: dict, registry: Optional[dict] = None) -> list[str]:
    """본문에 인용됐지만 수집된 출처에 없는 식별자를 찾는다.

    본문 인용 표기를 `[src_xxx]` 형태로 통일했을 때만 의미가 있다.
    """
    import re

    cited = set(re.findall(r"\[(src_[A-Za-z0-9_]+)\]", report_md))
    known = {str(src.get("id")) for src in collect_sources(state, registry)}
    return sorted(cited - known)


def find_uncited_sources(report_md: str, state: dict, registry: Optional[dict] = None) -> list[str]:
    """수집됐지만 본문에서 한 번도 인용되지 않은 출처를 찾는다.

    과제는 "실제로 활용한 자료 목록만" 기재하라고 했으므로 이 목록은 REFERENCE에서 빼야 한다.
    """
    known = collect_sources(state, registry)
    return sorted(
        str(src.get("id"))
        for src in known
        if str(src.get("id")) not in report_md and str(src.get("url") or "") not in report_md
    )


if __name__ == "__main__":
    from tests.mock_state import MOCK_STATE, SOURCE_REGISTRY

    print("=== REFERENCE ===")
    print(render_references(MOCK_STATE, SOURCE_REGISTRY))
    print()
    print("=== 점검 ===")
    report = audit_sources(MOCK_STATE, SOURCE_REGISTRY)
    print(f"수집된 출처: {report['total']}건")
    print(f"표기 불완전: {len(report['incomplete'])}건")
    for src in report["incomplete"]:
        print(f"  - {src.get('id')}")
    print(f"출처 아님(제외): {report['dropped']}")
