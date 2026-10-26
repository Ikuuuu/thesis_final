"""(선택) LLM 기반 관계 추출 — 규칙 기반 추출기와의 교차검증용.

OpenAI 호환 Chat Completions API(vLLM, Ollama, OpenAI 등)에 조문을 보내 인용·위임 관계를 JSON 으로 받는다.
외부 라이브러리 없이 표준 라이브러리(urllib)만 사용하므로 폐쇄망 vLLM 서버에도 그대로 붙일 수 있다.

    export LLM_BASE_URL=http://localhost:8000/v1     # vLLM 서버
    export LLM_MODEL=Qwen2.5-7B-Instruct
    python scripts/compare_llm.py --n 50
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Dict, List

SYSTEM_PROMPT = """너는 한국 법령 문서에서 법령 간 관계를 추출하는 정보추출기다.
주어진 조문에서 다른 법령을 가리키는 표현을 모두 찾아 JSON 배열로만 답하라.
각 원소 형식: {"target": "법령 정식명칭", "article_no": "조문번호(없으면 빈 문자열, 예: 5, 5의2)", "relation": "인용" 또는 "위임"}
- "위임" : 이 조문이 해당 법령에서 위임받은 사항을 정한다고 밝히는 경우 (…에서 위임한 사항, …에 따라 …을 정함)
- "인용" : 그 밖에 해당 법령·조문을 참조하는 경우
- 자기 자신(현재 법령)은 제외한다. 설명 없이 JSON 배열만 출력한다."""


def build_messages(law_title: str, article_no: str, article_text: str) -> List[Dict[str, str]]:
    user = f"[현재 법령] {law_title}\n[조문] 제{article_no}조\n[본문]\n{article_text.strip()}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_llm_json(text: str) -> List[Dict[str, str]]:
    """LLM 응답에서 JSON 배열을 관대하게 파싱한다 (코드블록·앞뒤 설명 허용)."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and it.get("target"):
            out.append({"target": str(it["target"]).strip(),
                        "article_no": str(it.get("article_no", "") or "").strip(),
                        "relation": it.get("relation", "인용")})
    return out


def call_llm(messages, base_url=None, model=None, api_key=None, timeout=60) -> str:
    base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
    model = model or os.environ.get("LLM_MODEL", "Qwen2.5-7B-Instruct")
    api_key = api_key or os.environ.get("LLM_API_KEY", "EMPTY")
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.0}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_with_llm(law_title: str, article_no: str, article_text: str, **kw) -> List[Dict[str, str]]:
    return parse_llm_json(call_llm(build_messages(law_title, article_no, article_text), **kw))
