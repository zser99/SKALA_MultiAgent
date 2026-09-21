"""1. 기술 선정 노드.

우선순위:
1) state에 selected_sw / selected_hw 가 이미 있으면 그대로 통과 (수동 override)
2) state["auto_select"] = True 이면 LLM이 Doc Pool에서 직접 선정 (에이전트 기반, 1안)
3) 기본값: 팀 최종 결론(candidates.FINAL_SELECTION)을 사용 — SW=KIVI, HW=ITME
   (Human 기반 선정, 2안. 이해 난이도·자료 확보 용이성·1.5일 작업 부담을 비교해 확정함)
"""
from llm import get_llm
from candidates import SW_POOL, HW_POOL, FINAL_SELECTION, FINAL_SELECTION_RATIONALE
from agents.utils import load_prompt, parse_json_response


def _pool_to_text(pool):
    return "\n".join(
        f"- id={c['id']} | {c['title']} ({c['year']}): {c['summary']}" for c in pool
    )


def _find(pool, cid):
    return next(c for c in pool if c["id"] == cid)


def tech_selection_node(state: dict) -> dict:
    sw_pool = state.get("sw_pool") or SW_POOL
    hw_pool = state.get("hw_pool") or HW_POOL

    if state.get("selected_sw") and state.get("selected_hw"):
        return {
            "selection_rationale": state.get(
                "selection_rationale", "(사용자가 직접 선정한 기술)"
            )
        }

    if not state.get("auto_select"):
        return {
            "sw_pool": sw_pool,
            "hw_pool": hw_pool,
            "selected_sw": _find(sw_pool, FINAL_SELECTION["sw"]),
            "selected_hw": _find(hw_pool, FINAL_SELECTION["hw"]),
            "selection_rationale": FINAL_SELECTION_RATIONALE,
        }

    # auto_select=True 인 경우에만 에이전트 기반 선정(1안) 사용
    prompt = load_prompt("tech_selection").format(
        sw_candidates=_pool_to_text(sw_pool),
        hw_candidates=_pool_to_text(hw_pool),
    )
    resp = get_llm(temperature=0).invoke(prompt)
    result = parse_json_response(resp.content)

    return {
        "sw_pool": sw_pool,
        "hw_pool": hw_pool,
        "selected_sw": _find(sw_pool, result["selected_sw_id"]),
        "selected_hw": _find(hw_pool, result["selected_hw_id"]),
        "selection_rationale": result["rationale"],
    }
