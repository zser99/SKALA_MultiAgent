"""Assess SW/HW fit for the selected domain across five evidence-backed criteria."""
import json
from pathlib import Path

from pydantic import BaseModel, Field

from agents.utils import load_prompt
from llm import get_llm
from rag.ingest import build_vectorstore
from rag.retriever import search

DEFAULT_DOMAIN = "Cloud/Data Center 환경의 Long-context LLM Serving"
METRICS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "domain_metrics.json"
DOMAIN_REFERENCES = {
    "serving": {
        "id": "domain_pagedattention",
        "file": "PagedAttention.pdf",
        "title": "PagedAttention",
    },
    "quality": {
        "id": "domain_longbench",
        "file": "LongBench.pdf",
        "title": "LongBench",
    },
    "latency": {
        "id": "domain_distserve",
        "file": "DistServe.pdf",
        "title": "DistServe",
    },
}


def _load_metrics() -> dict:
    with METRICS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class Assessment(BaseModel):
    claim: str = Field(description="Korean finding; distinguish measurement from inference")
    source_ids: list[str] = Field(description="IDs of excerpts directly supporting the claim")
    conditions: str = Field(description="Workload, hardware, model, context length, or 'not stated'")
    limitation: str = Field(description="What cannot be inferred from this evidence")
    domain_context: str = Field(default="", description="Domain evaluation context, not a technology result")
    domain_source_ids: list[str] = Field(default_factory=list, description="IDs of domain reference excerpts")


def _missing_evidence() -> dict:
    return {
        "claim": "확인된 근거 없음",
        "source_ids": [],
        "conditions": "확인 불가",
        "limitation": "관련 근거를 확인하지 못함",
        "domain_context": "",
        "domain_source_ids": [],
    }


def _summary(criteria: dict, side: str) -> str:
    lines = []
    for criterion, sides in criteria.items():
        item = sides[side]
        detail = item["claim"]
        if item["source_ids"]:
            detail += f" (근거: {', '.join(item['source_ids'])})"
            if item["conditions"]:
                detail += f"; 조건: {item['conditions']}"
            if item["limitation"]:
                detail += f"; 한계: {item['limitation']}"
        else:
            detail += " (확인 불가)"
        lines.append(f"- {criterion}: {detail}")
    return "\n".join(lines)


def domain_node(state: dict) -> dict:
    domain = state.get("domain") or DEFAULT_DOMAIN
    metric_specs = _load_metrics()
    transform = state.get("retry_count", 0) > 0
    candidates = {"sw": state["selected_sw"], "hw": state["selected_hw"]}
    tech_research = state["tech_research"]
    if any(side not in tech_research for side in candidates):
        raise ValueError("domain_node requires tech_research results for both sw and hw")
    vectorstores = {
        side: build_vectorstore(candidate["id"], candidate["file"])[0]
        for side, candidate in candidates.items()
    }
    domain_vectorstores = {
        name: build_vectorstore(reference["id"], reference["file"])[0]
        for name, reference in DOMAIN_REFERENCES.items()
    }
    judge = get_llm(temperature=0).with_structured_output(Assessment)
    prompt_template = load_prompt("domain")
    analysis = {
        "domain": domain,
        "technologies": {side: candidate["title"] for side, candidate in candidates.items()},
        "upstream_research_sources": {
            side: tech_research[side].get("sources", []) for side in candidates
        },
        "criteria": {},
        "domain_context": {},
        "evidence_gaps": [],
        "comparison_caution": "서로 다른 실험 설정의 수치를 직접 우열 비교하지 않음",
    }
    sources = {}

    for criterion, spec in metric_specs.items():
        analysis["criteria"][criterion] = {}
        description = spec["query"]
        reference_name = criterion if criterion in ("quality", "latency") else "serving"
        reference = DOMAIN_REFERENCES[reference_name]
        reference_store = domain_vectorstores[reference_name]
        domain_docs = search(
            reference_store, f"{domain} {description}", k=5 if transform else 3,
            transform=transform, title=reference["title"],
        )
        domain_excerpts = []
        for index, doc in enumerate(domain_docs):
            sid = f"{reference['id']}-{criterion}-{index}"
            metadata = doc.metadata
            page = metadata.get("page")
            sources[sid] = {
                "id": sid,
                "role": "domain_reference",
                "title": reference["title"],
                "source": str(metadata.get("source_file") or metadata.get("source") or reference["file"]),
                "page": page + 1 if isinstance(page, int) else page,
                "criterion": criterion,
            }
            domain_excerpts.append({"id": sid, "text": doc.page_content[:2500]})
        for side, candidate in candidates.items():
            prior_report = tech_research[side]
            query = (
                f"{candidate['title']} {domain} {description} "
                f"{prior_report.get('approach', '')[:240]} experiment result tradeoff"
            )
            docs = search(
                vectorstores[side], query, k=8 if transform else 5,
                transform=transform, title=candidate["title"],
            )
            excerpts = []
            for index, doc in enumerate(docs):
                sid = f"{candidate['id']}-{criterion}-{index}"
                metadata = doc.metadata
                page = metadata.get("page")
                sources[sid] = {
                    "id": sid,
                    "role": "technology",
                    "technology": candidate["title"],
                    "title": candidate["title"],
                    "source": str(metadata.get("source_file") or metadata.get("source") or candidate["file"]),
                    "url": candidate.get("url", ""),
                    "page": page + 1 if isinstance(page, int) else page,
                    "criterion": criterion,
                }
                excerpts.append({"id": sid, "text": doc.page_content[:2500]})

            if excerpts or domain_excerpts:
                result = judge.invoke([
                    ("system", prompt_template.format(
                        title=candidate["title"],
                        domain=domain,
                        criterion=criterion,
                        definition=spec["definition"],
                        items=", ".join(spec["items"]),
                        question=spec["question"],
                        design_reference=spec["design_reference"]["title"],
                    )),
                    ("human", json.dumps({
                        "technology": candidate["title"],
                        "domain": domain,
                        "criterion": criterion,
                        "description": description,
                        "prior_tech_research": prior_report,
                        "technology_excerpts": excerpts,
                        "domain_excerpts": domain_excerpts,
                    }, ensure_ascii=False, default=str)),
                ])
                entry = result.model_dump()
                valid_ids = {excerpt["id"] for excerpt in excerpts}
                entry["source_ids"] = list(dict.fromkeys(
                    sid for sid in entry["source_ids"] if sid in valid_ids
                ))
                valid_domain_ids = {excerpt["id"] for excerpt in domain_excerpts}
                entry["domain_source_ids"] = list(dict.fromkeys(
                    sid for sid in entry["domain_source_ids"] if sid in valid_domain_ids
                ))
                if not entry["source_ids"]:
                    entry["claim"] = "확인된 근거 없음"
                    entry["conditions"] = "확인 불가"
                    entry["limitation"] = "기술 논문에서 관련 근거를 확인하지 못함"
                if not entry["domain_source_ids"]:
                    entry["domain_context"] = ""
            else:
                entry = _missing_evidence()

            if not entry["source_ids"]:
                analysis["evidence_gaps"].append({"technology": side, "criterion": criterion})
            if entry["domain_source_ids"] and criterion not in analysis["domain_context"]:
                analysis["domain_context"][criterion] = {
                    "claim": entry["domain_context"],
                    "source_ids": entry["domain_source_ids"],
                }
            entry.pop("domain_context")
            entry.pop("domain_source_ids")
            analysis["criteria"][criterion][side] = entry

        analysis["domain_context"].setdefault(criterion, {"claim": "", "source_ids": []})

    used_ids = {
        sid for sides in analysis["criteria"].values()
        for entry in sides.values()
        for sid in entry["source_ids"]
    }
    used_ids.update(
        sid for context in analysis["domain_context"].values()
        for sid in context["source_ids"]
    )
    used_sources = [source for sid, source in sources.items() if sid in used_ids]
    memory_context = analysis["domain_context"]["memory_efficiency"]
    selection_rationale = {
        "domain": domain,
        "problem": memory_context["claim"],
        "source_ids": memory_context["source_ids"],
        "comparison_scope": analysis["technologies"],
    }
    output = {
        "domain": domain,
        "domain_result": {
            "sw": _summary(analysis["criteria"], "sw"),
            "hw": _summary(analysis["criteria"], "hw"),
            "evaluation_framework": metric_specs,
            "selection_rationale": selection_rationale,
            "domain_analysis": analysis,
            "domain_sources": used_sources,
        },
    }
    output["domain_source"] = list(dict.fromkeys(
        f"{source.get('url') or source['source']}#page={source['page']}"
        if source["page"] is not None else source.get("url") or source["source"]
        for source in used_sources
    ))
    return output
