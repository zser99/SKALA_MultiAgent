"""과제 문서의 'Doc Pool'을 그대로 코드화한 후보 목록.

각 항목의 `file` 은 data/ 폴더에 사용자가 직접 내려받아 넣어야 하는 PDF 파일명이다.
(arXiv 등 외부 다운로드는 이 실행 환경에서 직접 수행하지 않았으므로, 실제 실행 전
아래 url에서 PDF를 받아 지정된 파일명으로 data/ 에 넣어줘야 RAG가 정상 동작한다.)
"""
from state import TechCandidate

# 팀 최종 결론 (Human 기반 선정, 1.5일 부담·자료 확보 용이성 비교 후 확정):
#   SW = KIVI   — 이해 난이도 낮음~중간, 공개자료 많음, RAG/시장/도메인 자료 확보 쉬움
#   HW = ITME   — 최신 CXL 기반 계층형 메모리 확장, 도메인(Cloud/DC) 평가 적합성 매우 높음
FINAL_SELECTION = {"sw": "kivi", "hw": "itme"}
FINAL_SELECTION_RATIONALE = (
    "SW 진영은 KIVI를 선정했다. 양자화 계열의 대표적인 베이스라인으로 공개 자료가 많고 "
    "구현체(CUDA)가 이미 존재해 RAG 문서 구성과 시장·도메인 자료 확보가 용이하다. "
    "HW 진영은 ITME를 선정했다. CXL-Hybrid 메모리 기반의 최신 계층적 확장 아키텍처로, "
    "Cloud/Data Center의 Long-context LLM Serving 도메인에서의 적용성 평가가 매우 뚜렷하다. "
    "두 기술은 '데이터를 작게 만들자(SW)'와 '담을 공간을 넓히자(HW)'라는 상반된 접근을 "
    "대표하면서도, 평가에 필요한 공개 정보 수준이 비교적 균형 잡혀 있어 다관점 비교가 가능하다."
)

SW_POOL: list[TechCandidate] = [
    {
        "id": "turboquant",
        "camp": "SW",
        "title": "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
        "year": "2025-04",
        "summary": "KV cache를 3비트로 양자화하여 어텐션 처리 속도를 최대 8배 향상.",
        "url": "https://arxiv.org/pdf/2504.19874",
        "file": "sw_turboquant.pdf",
    },
    {
        "id": "deepseek_v2",
        "camp": "SW",
        "title": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
        "year": "2024-06",
        "summary": "MLA(Multi-head Latent Attention)로 KV를 저차원 잠재공간에 압축, KV cache 93.3% 감소.",
        "url": "https://arxiv.org/pdf/2405.04434",
        "file": "sw_deepseek_v2.pdf",
    },
    {
        "id": "kivi",
        "camp": "SW",
        "title": "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
        "year": "2024-06-25",
        "summary": "양자화 계열의 대표적인 베이스라인 기법 (튜닝 불필요, 비대칭 2비트 양자화).",
        "url": "https://arxiv.org/pdf/2402.02750",
        "file": "sw_kivi.pdf",
    },
]

HW_POOL: list[TechCandidate] = [
    {
        "id": "infinigen",
        "camp": "HW",
        "title": "InfiniGen: Efficient Generative Inference of LLMs with Dynamic KV Cache Management",
        "year": "2024-06",
        "summary": "필수 KV 항목만 프리패치하여 호스트 메모리 오프로딩 성능 최대 3배 개선.",
        "url": "https://arxiv.org/pdf/2406.19707",
        "file": "hw_infinigen.pdf",
    },
    {
        "id": "itme",
        "camp": "HW",
        "title": "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories",
        "year": "2026-06",
        "summary": "CXL-Hybrid 메모리 기반 계층적 메모리 확장 아키텍처, 처리량 1.80배 향상.",
        "url": "https://arxiv.org/pdf/2606.12556",
        "file": "hw_itme.pdf",
    },
    {
        "id": "pnm_cxl",
        "camp": "HW",
        "title": "Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits",
        "year": "2025-10",
        "summary": "DIMM 기반 PIM으로 용량과 대역폭을 동시에 확장.",
        "url": "https://arxiv.org/pdf/2511.00321",
        "file": "hw_pnm_cxl.pdf",
    },
]
