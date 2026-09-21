import json
import os
import re

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def load_prompt(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, f"{name}.txt"), encoding="utf-8") as f:
        return f.read()


def parse_json_response(text: str) -> dict:
    """LLM이 코드블록(```json ... ```)으로 감싸는 경우까지 방어적으로 파싱."""
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
