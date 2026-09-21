"""실행 스크립트.

Usage:
    python app.py
    python app.py --domain "OnDevice AI"
"""
import argparse
import datetime
import os

from graph import build_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        default=None,
        help="평가 대상 도메인 (예: '데이터센터/클라우드 서빙', 'OnDevice AI', '장문맥 처리 어플리케이션')",
    )
    args = parser.parse_args()

    app = build_graph()

    initial_state = {}
    if args.domain:
        initial_state["domain_focus"] = args.domain

    print("[graph] 실행 시작...")
    final_state = app.invoke(initial_state)

    if final_state.get("warnings"):
        print("\n[warnings]")
        for w in final_state["warnings"]:
            print(" -", w)

    os.makedirs("outputs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("outputs", f"report_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_state["final_report"])

    print(f"\n[done] 보고서 저장: {out_path}")
    print(f" - 선정 SW: {final_state['selected_sw']['title']}")
    print(f" - 선정 HW: {final_state['selected_hw']['title']}")


if __name__ == "__main__":
    main()
