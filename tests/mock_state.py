"""보고서 생성 에이전트 단독 개발/테스트용 Mock State.

보고서 노드는 그래프의 마지막에 있어서, 앞의 5개 에이전트가 모두 끝나야 실제 입력이
생긴다. 그때까지 기다리면 개발 시간이 없으므로 이 파일의 가짜 State로 먼저 개발한다.

    from tests.mock_state import MOCK_STATE
    from agents.report import report_node
    print(report_node(MOCK_STATE)["final_report"])

이 파일은 각 에이전트가 무엇을 채워야 하는지에 대한 **출력 스펙 계약**도 겸한다.
담당자는 자기 노드의 반환값이 아래 형태와 맞는지 확인할 것.

주의: 아래 내용은 전부 형식 확인용 예시 데이터이며 검증된 사실이 아니다.
"""

# --- 기술 선정 (agents/tech_selection.py) -----------------------------------
_SW = {
    "id": "kivi",
    "camp": "SW",
    "title": "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
    "year": "2024-02",
    "summary": "Key는 per-channel, Value는 per-token으로 2비트 양자화해 KV cache 용량을 줄인다.",
    "url": "https://arxiv.org/abs/2402.02750",
    "file": "sw_kivi.pdf",
}

_HW = {
    "id": "itme",
    "camp": "HW",
    "title": "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories",
    "year": "2026-06",
    "summary": "CXL 하이브리드 메모리를 계층으로 붙여 TB급 KV cache를 수용한다.",
    "url": "https://arxiv.org/abs/2606.12556",
    "file": "hw_itme.pdf",
}

_SELECTION_RATIONALE = (
    "SW 진영은 KIVI, HW 진영은 ITME를 선정했다. 동일한 KV cache 메모리 병목에 대해 "
    "KIVI는 저장할 데이터의 크기를 줄이고 ITME는 사용 가능한 저장 공간을 확장한다. "
    "문제는 같지만 해결 계층과 방식이 상반되어 관점별 평가 차이가 드러나기 쉽다."
)

# --- 기술 조사 (agents/tech_research.py) ------------------------------------
_TECH_RESEARCH = {
    "sw": {
        "overview": "KV cache를 2비트로 양자화하는 튜닝 불필요(tuning-free) 기법.",
        "approach": "Key는 채널 단위, Value는 토큰 단위로 비대칭 양자화한다. "
                    "Key의 이상치가 특정 채널에 몰리는 반면 Value는 그렇지 않다는 관찰에 기반한다.",
        "limitations": "손실 압축이므로 출력 품질 저하 가능성이 있고, 양자화/역양자화 "
                       "오버헤드가 지연시간에 더해진다.",
        "trl": "논문과 공개 구현체, 재현 벤치마크가 존재한다(TRL 4~5 추정). "
               "주류 프레임워크의 KV cache 양자화 구현에 영향을 준 것으로 보고되나, "
               "상용 운영 사례는 공개 정보에서 확인되지 않았다.",
        "sources": ["https://arxiv.org/abs/2402.02750"],
    },
    "hw": {
        "overview": "CXL 하이브리드 메모리를 계층으로 연결해 KV cache 용량을 확장하는 아키텍처.",
        "approach": "NVMe SSD에서 CXL 하이브리드 메모리 내부 DRAM 캐시로 하드웨어 레벨 "
                    "프리패치를 수행하고, 다단 DMA 파이프라인을 GPU 연산과 겹쳐 지연을 가린다.",
        "limitations": "CXL 인프라 도입 비용과 조달 기간이 필요하고, 원격 메모리 접근에 따른 "
                       "지연시간이 남는다. 공개 성능 수치가 제한적이다.",
        "trl": "2026-06 공개 논문으로 프로토타입 수준 검증이 보고된다(TRL 4~6 추정). "
               "다만 기반 플랫폼인 CXL 자체는 상용 제품이 출시된 단계로, "
               "기술과 플랫폼의 성숙도 층위가 다르다.",
        "sources": ["https://arxiv.org/abs/2606.12556"],
    },
}

# --- 병렬 관점 평가 (market / stakeholder / domain) --------------------------
# NOTE: 현재 market 노드는 sources에 "sw_basis=..." 같은 라벨을 넣고,
#       stakeholder/domain 노드는 sources를 아예 반환하지 않는다.
#       REFERENCE를 제대로 만들려면 아래 SOURCE_REGISTRY 형태가 필요하다.
_MARKET = {
    "sw": "LLM 추론 비용 최적화 수요를 배경으로 KV cache 양자화에 대한 관심이 이어지고 있다. "
          "오픈소스 구현이 공개되어 있어 프레임워크 차원의 도입 장벽이 낮은 편이다.",
    "hw": "AI 데이터센터용 메모리 확장 수요와 CXL 생태계 형성이 맞물려 있다. "
          "메모리 업계의 제품 로드맵과 컨소시엄 활동이 확인되나, ITME 자체의 채택 사례는 확인되지 않았다.",
    "summary": "SW는 오픈소스 생태계 중심, HW는 벤더·표준 생태계 중심으로 시장 근거의 성격이 다르다.",
    "sources": ["src_market_01", "src_market_02"],
}

_STAKEHOLDER = {
    "sw": "서빙 개발자는 코드 수정만으로 적용 가능한 점을 이점으로 보지만, "
          "품질 저하 허용 범위를 도입 장벽으로 지적한다.",
    "hw": "클라우드 사업자는 GPU 증설 대비 TCO 개선 가능성에 관심을 보이나, "
          "서버 OEM과 메모리 업체의 공급망 준비 수준을 전제 조건으로 본다.",
    "summary": "개발자는 도입 용이성을, 인프라 사업자는 투자 회수 구조를 기준으로 평가한다.",
    "sources": ["src_stake_01"],
}

_DOMAIN = {
    "sw": "장문맥 서빙에서 메모리 부담을 직접 낮춰 배치 크기를 키울 수 있으나, "
          "품질 민감도가 높은 워크로드에서는 적용 범위가 제한된다.",
    "hw": "대규모 배치 추론 환경에서 용량 한계를 물리적으로 해소하지만, "
          "기존 데이터센터에 적용하려면 하드웨어 조달과 검증 기간이 필요하다.",
    "summary": "동일 도메인 안에서도 품질 민감도와 인프라 투자 여력에 따라 평가가 갈린다.",
    "sources": ["src_domain_01", "src_domain_02"],
}

# --- 평가 종합 (agents/synthesis.py) ----------------------------------------
_SYNTHESIS = {
    "agreements": [
        "두 기술 모두 장문맥 서빙에서 KV cache가 병목이라는 문제 인식은 동일하다.",
        "두 기술 모두 공개 정보만으로는 상용 운영 단계 검증을 확인하기 어렵다.",
    ],
    "conflicts": [
        {
            "claim_sw": "SW 압축은 코드 수정만으로 즉시 적용 가능해 도입 속도가 빠르다.",
            "claim_hw": "HW 확장은 조달·검증 기간이 필요해 도입이 느리다.",
            "type": "시점차",
        },
        {
            "claim_sw": "품질 저하 가능성이 있어 정확도 민감 워크로드에서는 채택이 어렵다.",
            "claim_hw": "정보를 유지하므로 품질 영향은 없으나 인프라 비용이 발생한다.",
            "type": "측정기준차",
        },
    ],
    "implications": [
        "시장성 관점에서는 생태계 자료가 풍부한 쪽이 유리해 보이지만, 이는 근거 가용성 차이일 수 있다.",
        "도메인 관점에서는 품질 민감도가 평가를 뒤집는 변수로 작용한다.",
    ],
}

# --- 최종 Mock State ---------------------------------------------------------
MOCK_STATE = {
    "sw_pool": [_SW],
    "hw_pool": [_HW],
    "domain": "Cloud/Data Center 환경의 Long-context LLM Serving",
    "selected_sw": _SW,
    "selected_hw": _HW,
    "selection_rationale": _SELECTION_RATIONALE,
    "tech_research": _TECH_RESEARCH,
    "market_result": _MARKET,
    "stakeholder_result": _STAKEHOLDER,
    "domain_result": _DOMAIN,
    "evidence_sufficient": True,
    "retry_count": 1,
    "synthesis": _SYNTHESIS,
    "final_report": "",
    "warnings": [],
}


# --- 팀에 제안할 출처 스펙 ----------------------------------------------------
# 과제 REFERENCE는 특허/논문/웹 표기 형식이 각각 다르고 "실제 활용한 자료만" 기재해야 한다.
# 지금처럼 sources가 URL 문자열이면 유형 구분이 불가능하므로, 각 에이전트가 아래 형태로
# 반환해 주면 REFERENCE를 코드로 정확히 생성할 수 있다.
#
#   paper  : {"id", "type": "paper",  "authors", "year", "title", "venue", "url"}
#   patent : {"id", "type": "patent", "applicant", "date", "title", "number", "url"}
#   web    : {"id", "type": "web",    "org", "date", "title", "site", "url"}
#
# id는 본문 인용 표기(예: [src_market_01])와 대조하는 데 쓴다.
SOURCE_REGISTRY = {
    "src_kivi": {
        "id": "src_kivi",
        "type": "paper",
        "authors": "Liu, Z., Yuan, J., Jin, H., et al.",
        "year": "2024",
        "title": "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
        "venue": "ICML 2024",
        "url": "https://arxiv.org/abs/2402.02750",
    },
    "src_itme": {
        "id": "src_itme",
        "type": "paper",
        "authors": "(저자 확인 필요)",
        "year": "2026",
        "title": "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories",
        "venue": "arXiv preprint arXiv:2606.12556",
        "url": "https://arxiv.org/abs/2606.12556",
    },
    "src_market_01": {
        "id": "src_market_01",
        "type": "web",
        "org": "(기관명 확인 필요)",
        "date": "2026-01-01",
        "title": "(시장 리포트 제목 확인 필요)",
        "site": "(사이트명)",
        "url": "https://example.com/market-report",
    },
    "src_market_02": {
        "id": "src_market_02",
        "type": "web",
        "org": "(기관명 확인 필요)",
        "date": "2026-02-01",
        "title": "(산업 기사 제목 확인 필요)",
        "site": "(사이트명)",
        "url": "https://example.com/industry-news",
    },
    "src_stake_01": {
        "id": "src_stake_01",
        "type": "web",
        "org": "(기관명 확인 필요)",
        "date": "2026-03-01",
        "title": "(이해관계자 발언 출처 확인 필요)",
        "site": "(사이트명)",
        "url": "https://example.com/stakeholder",
    },
    "src_domain_01": {
        "id": "src_domain_01",
        "type": "web",
        "org": "(기관명 확인 필요)",
        "date": "2026-04-01",
        "title": "(도메인 자료 제목 확인 필요)",
        "site": "(사이트명)",
        "url": "https://example.com/domain-a",
    },
    "src_domain_02": {
        "id": "src_domain_02",
        "type": "patent",
        "applicant": "(출원인 확인 필요)",
        "date": "2025-11",
        "title": "(특허명 확인 필요)",
        "number": "KR-10-2025-0000000",
        "url": "https://example.com/patent",
    },
}
