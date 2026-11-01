"""LLM-as-a-judge: rubric prompts, score parsing, and a minimal Ollama client."""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Optional

DEFAULT_RUBRIC = """Score the RESPONSE from 1 to 5:
5 - fully correct and complete, clearly explained, consistent with the reference
4 - correct final result, minor gaps or sloppiness in the explanation
3 - partially correct: right approach but an error, or correct but unjustified
2 - mostly incorrect, with some relevant steps
1 - wrong, irrelevant, or missing"""


def build_judge_prompt(question: str, response: str, reference: Optional[str] = None,
                       rubric: str = DEFAULT_RUBRIC) -> str:
    ref = f"REFERENCE ANSWER:\n{reference}\n\n" if reference else ""
    return (
        "You are a strict grader. Read the question, the reference (if given) and the "
        "response, then grade the response using the rubric.\n\n"
        f"RUBRIC:\n{rubric}\n\nQUESTION:\n{question}\n\n{ref}RESPONSE:\n{response}\n\n"
        "First write one or two sentences of justification. Then output the final line "
        "exactly as: SCORE: <integer 1-5>"
    )


def parse_judge_score(text: str, lo: int = 1, hi: int = 5) -> Optional[int]:
    """Prefer an explicit 'SCORE: k' line; fall back to the last in-range integer."""
    m = re.findall(r"SCORE\s*[:=]\s*(\d+)", text or "", flags=re.I)
    cands = m or re.findall(r"\b(\d+)\b", text or "")
    for c in reversed(cands):
        v = int(c)
        if lo <= v <= hi:
            return v
    return None


def ollama_chat(prompt: str, model: str = "llama3.1", url: str = "http://localhost:11434/api/chat",
                temperature: float = 0.0, seed: int = 123, timeout: float = 120.0) -> str:
    """Query a locally running Ollama server (``ollama serve``) without extra dependencies."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 4096},
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["message"]["content"]


def judge(question, response, reference=None, chat_fn=ollama_chat, **kw) -> Optional[int]:
    return parse_judge_score(chat_fn(build_judge_prompt(question, response, reference), **kw))
