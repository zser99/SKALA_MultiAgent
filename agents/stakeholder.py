"""3-b. 이해관계자 평가 에이전트 (RAG: X) — 병렬 실행되는 노드 중 하나.

벡터스토어(RAG) 대신 웹 검색 도구 + LLM 일반 지식을 사용한다.
DuckDuckGo 검색은 키가 필요 없어 기본값으로 넣었으나, 실행 환경에 인터넷이 제한되면
자동으로 빈 컨텍스트로 폴백해 파이프라인이 끊기지 않도록 했다.
"""
from llm import get_llm
from agents.utils import load_prompt, parse_json_response

try:
    from langchain_community.tools import DuckDuckGoSearchRun

    _search_tool = DuckDuckGoSearchRun()
except Exception:  # 패키지 미설치 또는 네트워크 제한 환경
    _search_tool = None


def _search(query: str) -> str:
    if _search_tool is None:
        return ""
    try:
        return _search_tool.invoke(query)
    except Exception:
        return ""


def _stakeholder_one(candidate: dict) -> str:
    search_context = _search(f"{candidate['title']} industry reaction adoption")
    prompt = load_prompt("stakeholder").format(
        title=candidate["title"], search_context=search_context or "(검색 결과 없음, 일반 지식으로 서술)"
    )
    resp = get_llm(temperature=0.4).invoke(prompt)
    result = parse_json_response(resp.content)
    return result["summary"]


def stakeholder_node(state: dict) -> dict:
    sw_summary = _stakeholder_one(state["selected_sw"])
    hw_summary = _stakeholder_one(state["selected_hw"])
    return {"stakeholder_result": {"sw": sw_summary, "hw": hw_summary}}
