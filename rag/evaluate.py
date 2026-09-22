"""Retrieval 평가: Hit Rate@K + MRR@K + 정성 평가용 스니펫 덤프.

동일한 질의셋으로 두 전략을 비교한다.
  - Baseline  : Dense Retrieval + Top-K
  - Transform : Query Transformation 후 RRF 융합

정답 라벨은 chunk 단위 수작업 라벨링 대신 gold keyword 집합으로 대체한다.
검색된 chunk가 해당 질의의 gold keyword를 **모두** 포함하면 정답 근거로 간주하며,
키워드는 모두 원문 PDF에 실제로 등장하는지 확인해 선정했다.

Usage:
    python -m rag.evaluate
    python -m rag.evaluate --k 5
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from candidates import SW_POOL, HW_POOL
from rag.ingest import build_vectorstore
from rag.retriever import search

# 질의는 한국어, 문서는 영문 논문 -> cross-lingual retrieval 성능까지 함께 본다.
EVAL_SET = [
    {
        "doc": "kivi",
        "query": "KIVI는 Key 캐시와 Value 캐시를 각각 어떤 축 기준으로 양자화하나요?",
        "gold": ["per-channel", "per-token"],
    },
    {
        "doc": "kivi",
        "query": "KIVI가 사용하는 양자화 비트 수와 비대칭 양자화 방식은 무엇인가요?",
        "gold": ["2bit", "asymmetric"],
    },
    {
        "doc": "kivi",
        "query": "KIVI는 별도의 추가 학습이나 튜닝 없이 적용할 수 있나요?",
        "gold": ["tuning-free"],
    },
    {
        "doc": "kivi",
        "query": "KIVI에서 residual length와 group size는 어떤 역할을 하나요?",
        "gold": ["residual length", "group size"],
    },
    {
        "doc": "kivi",
        "query": "KIVI를 적용하면 배치 크기와 처리량은 어떻게 변하나요?",
        "gold": ["batch size", "throughput"],
    },
    {
        "doc": "kivi",
        "query": "KIVI의 정확도는 어떤 벤치마크로 평가했나요?",
        "gold": ["LongBench"],
    },
    {
        "doc": "kivi",
        "query": "Key 캐시에서 이상치(outlier)는 어떤 패턴으로 나타나나요?",
        "gold": ["outlier", "channel"],
    },
    {
        "doc": "itme",
        "query": "ITME는 어떤 인터커넥트 표준을 사용해 메모리를 확장하나요?",
        "gold": ["CXL"],
    },
    {
        "doc": "itme",
        "query": "ITME의 계층형(tiered) 메모리는 어떻게 구성되나요?",
        "gold": ["tier", "DRAM"],
    },
    {
        "doc": "itme",
        "query": "ITME는 GPU 메모리 용량 한계 문제를 어떻게 해결하나요?",
        "gold": ["GPU memory", "KV cache"],
    },
    {
        "doc": "itme",
        "query": "ITME의 처리량 향상 수치는 얼마인가요?",
        "gold": ["1.80"],
    },
    {
        "doc": "itme",
        "query": "ITME 프로토타입은 실제 하드웨어로 검증되었나요?",
        "gold": ["prototype"],
    },
    {
        "doc": "itme",
        "query": "ITME에서 PCIe 대역폭은 성능에 어떤 영향을 주나요?",
        "gold": ["PCIe", "bandwidth"],
    },
    {
        "doc": "itme",
        "query": "ITME는 분리형(disaggregated) 메모리를 어떻게 활용하나요?",
        "gold": ["disaggregated"],
    },
]

_CANDIDATES = {c["id"]: c for c in SW_POOL + HW_POOL}


def _is_hit(text: str, gold: list[str]) -> bool:
    low = text.lower()
    return all(g.lower() in low for g in gold)


def _first_hit_rank(docs, gold: list[str]) -> int:
    """정답 근거가 처음 등장한 순위(1-based). 없으면 0."""
    for rank, d in enumerate(docs, start=1):
        if _is_hit(d.page_content, gold):
            return rank
    return 0


def _run(stores: dict, k: int, transform: bool) -> list[dict]:
    rows = []
    for item in EVAL_SET:
        vs = stores.get(item["doc"])
        docs = search(
            vs,
            item["query"],
            k=k,
            transform=transform,
            title=_CANDIDATES[item["doc"]]["title"],
        )
        rows.append(
            {
                "query": item["query"],
                "gold": item["gold"],
                "rank": _first_hit_rank(docs, item["gold"]),
                "top1": docs[0].page_content.strip().replace("\n", " ")[:180] if docs else "",
            }
        )
    return rows


def _metrics(rows: list[dict]) -> tuple[float, float]:
    hits = [r for r in rows if r["rank"] > 0]
    hit_rate = len(hits) / len(rows)
    mrr = sum(1.0 / r["rank"] for r in hits) / len(rows)
    return hit_rate, mrr


def _report(k: int, base: list[dict], trans: list[dict]) -> str:
    b_hr, b_mrr = _metrics(base)
    t_hr, t_mrr = _metrics(trans)

    lines = [
        f"# RAG Retrieval 평가 (K={k})",
        "",
        f"- 질의 수: {len(EVAL_SET)}개 (한국어 질의 / 영문 논문)",
        "- 정답 판정: 검색된 chunk가 해당 질의의 gold keyword를 모두 포함하면 정답 근거로 간주",
        "",
        "## 종합",
        "",
        f"| 전략 | Hit Rate@{k} | MRR@{k} |",
        "| --- | --- | --- |",
        f"| Baseline (Dense Top-K) | {b_hr:.3f} | {b_mrr:.3f} |",
        f"| Query Transformation + RRF | {t_hr:.3f} | {t_mrr:.3f} |",
        f"| 변화량 | {t_hr - b_hr:+.3f} | {t_mrr - b_mrr:+.3f} |",
        "",
        "## 질의별 정답 순위 (0 = Top-K 내 미검색)",
        "",
        "| # | 질의 | gold keyword | Baseline | Transform |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, (b, t) in enumerate(zip(base, trans), start=1):
        lines.append(
            f"| {i} | {b['query']} | {', '.join(b['gold'])} | {b['rank']} | {t['rank']} |"
        )

    diffs = [
        (i, b, t)
        for i, (b, t) in enumerate(zip(base, trans), start=1)
        if b["rank"] != t["rank"]
    ]
    lines += ["", "## 정성 평가 (두 전략의 결과가 달라진 질의의 Top-1 스니펫)", ""]
    if not diffs:
        lines.append("두 전략의 검색 순위가 모든 질의에서 동일했습니다.")
    for i, b, t in diffs[:5]:
        lines += [
            f"### Q{i}. {b['query']}",
            f"- Baseline (rank={b['rank']}) Top-1: {b['top1']}",
            f"- Transform (rank={t['rank']}) Top-1: {t['top1']}",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Top-K (기본 5)")
    args = parser.parse_args()

    stores = {}
    for doc_id in {item["doc"] for item in EVAL_SET}:
        c = _CANDIDATES[doc_id]
        vs, warning = build_vectorstore(c["id"], c["file"])
        if warning:
            print("[warn]", warning)
        if vs is None:
            print(f"[skip] {doc_id}: 벡터스토어가 없어 평가할 수 없습니다.")
            return
        stores[doc_id] = vs

    print("[eval] Baseline (Dense Top-K) 실행...")
    base = _run(stores, args.k, transform=False)
    print("[eval] Query Transformation 실행...")
    trans = _run(stores, args.k, transform=True)

    report = _report(args.k, base, trans)
    print("\n" + report)

    os.makedirs("outputs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("outputs", f"rag_eval_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[done] 평가 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
