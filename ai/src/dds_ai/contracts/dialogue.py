"""ИИ-руководитель: разрешённые действия и реплики (D-026, D-030, C-04)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .common import Contract, ModelRef, NonEmptyStr


class SupervisorAction(StrEnum):
    LISTEN = "listen"
    CLARIFY = "clarify"
    READ_BACK = "read_back"
    CONFIRM_RECEIPT = "confirm_receipt"


class ScenarioCallState(StrEnum):
    """Сценарные состояния — не технический сбой."""

    BUSY = "busy"
    NO_ANSWER = "no_answer"
    DROPPED = "dropped"


class ReplyMode(StrEnum):
    LLM = "llm"
    FALLBACK = "fallback"


class Interlocutor(Contract):
    """Контракт собеседника: позже подключается канал оператора 112 (D-030)."""

    role_id: NonEmptyStr
    published_fact_ids: tuple[NonEmptyStr, ...] = ()
    allowed_actions: tuple[SupervisorAction, ...] = Field(min_length=1)


class SupervisorReply(Contract):
    action: SupervisorAction
    text: NonEmptyStr
    used_fact_ids: tuple[NonEmptyStr, ...] = ()
    mode: ReplyMode
    model_ref: ModelRef | None = None

    @model_validator(mode="after")
    def _mode(self) -> SupervisorReply:
        if self.mode is ReplyMode.LLM and self.model_ref is None:
            raise ValueError("llm reply requires model_ref")
        return self

    def check_against(self, interlocutor: Interlocutor) -> None:
        """Реплика не выходит за разрешённые действия и опубликованные факты (инвариант 3)."""
        if self.action not in interlocutor.allowed_actions:
            raise ValueError(f"action {self.action} is not allowed")
        extra = set(self.used_fact_ids) - set(interlocutor.published_fact_ids)
        if extra:
            raise ValueError(f"reply uses unpublished facts: {sorted(extra)}")
