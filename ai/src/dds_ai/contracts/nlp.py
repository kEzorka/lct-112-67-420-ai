"""NLP-извлечение признаков (6.11, D-016).

Вход: текст описания / SMS / транскрипт. Выход — только подсказка (`ML hints`) для
движка правил и оценивателя: top-k типов происшествия, теги, слоты, пропуски, противоречия.
Финальный тип, ЕКП и службы назначает исключительно детерминированный движок маршрутизации
(`mocks/routing.py`) — этот контракт в него не пишет и не может подменить его решение.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .common import Contract, ModelRef, NonEmptyStr


class IncidentTypeHint(Contract):
    """Один из top-k кандидатов типа происшествия."""

    incident_type: NonEmptyStr
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="Только если модель его отдаёт; у правил — None"
    )


class SlotState(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    MISSING = "missing"


class TextSpan(Contract):
    """Символьное смещение во входном тексте — не транскрипт с таймкодами (Evidence из
    common.py), т.к. NLP-извлечение работает и с текстом без временной разметки (SMS,
    описание)."""

    excerpt: NonEmptyStr
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _range(self) -> TextSpan:
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class ExtractedSlot(Contract):
    """Слот карточки, извлечённый из текста (адрес, число пострадавших, обстоятельства)."""

    name: NonEmptyStr
    state: SlotState
    value: str | None = None
    evidence: TextSpan | None = None

    @model_validator(mode="after")
    def _state(self) -> ExtractedSlot:
        if self.state is SlotState.KNOWN and self.value is None:
            raise ValueError("known slot requires value")
        if self.state is not SlotState.KNOWN and self.value is not None:
            raise ValueError(f"{self.state} slot must not carry a value")
        return self


class MissingInfoFlag(Contract):
    """Пропуск обязательных сведений."""

    slot: NonEmptyStr
    description: NonEmptyStr


class ContradictionFlag(Contract):
    """Противоречивое описание: минимум два несовместимых упоминания одного слота."""

    slot: NonEmptyStr
    description: NonEmptyStr
    evidence: tuple[TextSpan, ...] = Field(min_length=2)


class NlpExtraction(Contract):
    """ML hints: служб не назначает (инвариант из 4.2/6.11), confidence только у модели (инв. 5)."""

    top_k: tuple[IncidentTypeHint, ...] = ()
    tags: tuple[NonEmptyStr, ...] = ()
    slots: tuple[ExtractedSlot, ...] = ()
    missing: tuple[MissingInfoFlag, ...] = ()
    contradictions: tuple[ContradictionFlag, ...] = ()
    needs_clarification: bool = False
    model_ref: ModelRef | None = Field(
        default=None, description="None — правило (базовая реализация), иначе версия модели"
    )

    @model_validator(mode="after")
    def _rule_based_has_no_confidence(self) -> NlpExtraction:
        if self.model_ref is None and any(h.confidence is not None for h in self.top_k):
            raise ValueError("rule-based extraction must not carry a confidence value")
        return self
