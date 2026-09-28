"""Детерминированный фейковый LLMProvider — для тестов и самопроверки обвязки замера.

Не модель: возвращает черновик реплики из промпта в формате ответа модели. Задержка
генерации задаётся явно, чтобы обвязка замера показывала работу стадий без весов.
"""

from __future__ import annotations

import json
import re
import time

from ..contracts.common import ModelRef

_DRAFT = re.compile(r'<черновик действие="([a-z_]+)">\n(.*?)\n</черновик>', re.S)


def draft_of(prompt: str) -> tuple[str, str]:
    """Действие и черновик из собранного промпта (`prompting.PromptTemplate.build`)."""
    m = _DRAFT.search(prompt)
    if m is None:
        raise ValueError("prompt has no draft block")
    return m.group(1), m.group(2)


class EchoDraftLLM:
    model_ref = ModelRef(component="llm", model_name="echo-draft-fake", model_version="0")

    def __init__(self, latency_s: float = 0.0):
        self.latency_s = latency_s
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        self.prompts.append(prompt)
        if self.latency_s:
            time.sleep(self.latency_s)
        action, draft = draft_of(prompt)
        return json.dumps({"action": action, "text": draft}, ensure_ascii=False)
