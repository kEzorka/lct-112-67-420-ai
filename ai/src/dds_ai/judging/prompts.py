"""Сборка промпта семантического оценивателя (6.6). Версия промпта идёт в `ModelRef.prompt_version`
(инвариант 6). Эталон сравнения передаётся оценивателю намеренно (он выставляет балл, а не
разговаривает с обучаемым) — в отличие от промпта руководителя (инвариант 1), здесь это не
нарушение: сравнение не попадает обратно в реплики или в карточку, только в `CriterionResult`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[3] / "config"
DEFAULT_PROMPT = CONFIG / "judge.prompt.ru.json"

_MARKUP = re.compile(r"[<>{}`]")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class JudgePromptTemplate:
    version: str
    field_instructions: str
    conversation_instructions: str
    max_tokens_field: int
    max_tokens_conversation: int
    max_context_chars: int

    @classmethod
    def load(cls, path: Path = DEFAULT_PROMPT) -> JudgePromptTemplate:
        data = json.loads(path.read_text("utf-8"))
        return cls(
            version=data["prompt_version"],
            field_instructions=data["field_instructions"],
            conversation_instructions=data["conversation_instructions"],
            max_tokens_field=int(data["max_tokens_field"]),
            max_tokens_conversation=int(data["max_tokens_conversation"]),
            max_context_chars=int(data["max_context_chars"]),
        )

    def as_data(self, text: str) -> str:
        """Проверяемый текст — данные (инвариант 2): без разметки блоков, с ограничением длины."""
        clean = _SPACES.sub(" ", _MARKUP.sub(" ", text)).strip()
        if len(clean) > self.max_context_chars:
            clean = clean[: self.max_context_chars].rstrip() + " …"
        return clean

    def field_prompt(self, *, expected: str, student_text: str) -> str:
        return self.field_instructions.format(
            expected=self.as_data(expected), student_text=self.as_data(student_text)
        )

    def conversation_prompt(self, *, items: str, transcript: str) -> str:
        return self.conversation_instructions.format(
            items=self.as_data(items), transcript=self.as_data(transcript)
        )
