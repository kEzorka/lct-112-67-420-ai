"""Шов для LLM-формулировки реплик (M3) и проверка выхода, общая для LLM и шаблона.

На M1 генератор реплик — только шаблон. LLM, если подключена, лишь переформулирует уже
выбранную автоматом реплику; любой сбой или невалидный выход → шаблонная реплика (C-04).
В промпт не передаются эталон и проверяемые элементы доклада (инвариант 1).
"""

from __future__ import annotations

from uuid import UUID

from ..contracts.common import ModelRef
from ..contracts.dialogue import SupervisorAction
from ..contracts.events import ComponentName
from ..contracts.worker import Lane
from ..ports import LLMProvider
from ..worker import InferenceWorker
from .matching import digits

MAX_REPLY_CHARS = 300
PROMPT_VERSION = "supervisor-rephrase-m1-draft"


class InvalidReply(ValueError):
    pass


def validate_reply(text: str, allowed_context: str, *, max_chars: int = MAX_REPLY_CHARS) -> str:
    """Реплика не пустая, ограничена по длине и не содержит чисел вне контекста (6.4)."""
    text = text.strip()
    if not text:
        raise InvalidReply("empty reply")
    if len(text) > max_chars:
        raise InvalidReply(f"reply longer than {max_chars}")
    extra = digits(text) - digits(allowed_context)
    if extra:
        raise InvalidReply(f"reply contains numbers outside context: {sorted(extra)}")
    return text


class LLMPhraser:
    def __init__(self, llm: LLMProvider, worker: InferenceWorker, *, role_title: str):
        self.llm = llm
        self.worker = worker
        self.role_title = role_title

    @property
    def model_ref(self) -> ModelRef:
        ref = self.llm.model_ref
        return ref.model_copy(update={"prompt_version": PROMPT_VERSION})

    def prompt(self, action: SupervisorAction, draft: str) -> str:
        return (
            f"Ты — {self.role_title}. Действие: {action.value}.\n"
            "Переформулируй реплику коротко, по-деловому. Не добавляй новых фактов, "
            "чисел и адресов.\n"
            f"Реплика: {draft}"
        )

    def phrase(
        self,
        action: SupervisorAction,
        draft: str,
        allowed_context: str,
        *,
        attempt_id: UUID | None = None,
    ) -> str:
        def run() -> str:
            out = self.llm.complete(self.prompt(action, draft), max_tokens=120)
            return validate_reply(out, allowed_context)

        return self.worker.call(ComponentName.LLM, Lane.INTERACTIVE, run, attempt_id=attempt_id)
