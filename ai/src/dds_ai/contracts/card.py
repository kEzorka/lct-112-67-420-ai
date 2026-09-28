"""Карточка происшествия (D-011) — черновик контракта.

Состав полей D-011 фиксирует бэкенд по версии схемы; здесь — оболочка поля и карточки.
Эталон и учебные метаданные в карточку не входят (инвариант 1).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from .common import Contract, NonEmptyStr, VersionRef


class FieldState(StrEnum):
    """«Неизвестно», «нет» и «не применимо» не сводятся к пустой строке или нулю."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    NONE = "none"
    NOT_APPLICABLE = "not_applicable"


class FieldOrigin(StrEnum):
    OPERATOR_112 = "operator_112"
    DISPATCHER = "dispatcher"


class CardField(Contract):
    state: FieldState
    raw: Any = None
    normalized: Any = None
    origin: FieldOrigin
    visible_to_trainee: bool = True

    @model_validator(mode="after")
    def _state(self) -> CardField:
        if self.state is FieldState.KNOWN and self.raw is None:
            raise ValueError("known field requires raw value")
        if self.state is not FieldState.KNOWN and (
            self.raw is not None or self.normalized is not None
        ):
            raise ValueError(f"{self.state} field must not carry a value")
        return self


class CardGeneration(Contract):
    """Всё, что нужно для воспроизведения исходной карточки (D-003)."""

    scenario: VersionRef
    variation_params: dict[str, Any] = Field(default_factory=dict)
    seed: int
    model_ref_id: str | None = Field(default=None, description="Ссылка на ModelRef в снимке версий")


class IncidentCard(Contract):
    """Исходная карточка от ИИ-оператора 112. Неизменна; правки диспетчера — ревизии."""

    card_id: UUID
    card_schema: VersionRef
    created_at: AwareDatetime
    fields: dict[NonEmptyStr, CardField]
    generation: CardGeneration

    @model_validator(mode="after")
    def _origin(self) -> IncidentCard:
        if any(f.origin is not FieldOrigin.OPERATOR_112 for f in self.fields.values()):
            raise ValueError("source card fields must originate from operator_112")
        return self


class CardRevisionRecord(Contract):
    card_id: UUID
    revision: int = Field(ge=1)
    author_id: UUID
    created_at: AwareDatetime
    changes: dict[NonEmptyStr, CardField] = Field(min_length=1)
