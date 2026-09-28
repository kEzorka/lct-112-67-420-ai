"""Общие типы контрактов.

Совместимы с бэкендом (docs/backend/contract-notes.md): идентификаторы UUID,
время — ISO 8601 с часовым поясом (timestamptz), произвольные данные — JSON (jsonb).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyStr = Annotated[str, Field(min_length=1)]


class Contract(BaseModel):
    """База всех контрактов: неизвестные поля запрещены, объекты неизменяемы (C-09)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VersionRef(Contract):
    """Ссылка на версионируемый артефакт: сценарий, правила, рубрику, политику времени."""

    name: NonEmptyStr
    version: NonEmptyStr


class ModelRef(Contract):
    """Версия модели и промпта, на которой принято решение (инвариант 6)."""

    component: NonEmptyStr
    model_name: NonEmptyStr
    model_version: NonEmptyStr
    prompt_version: str | None = None


class EvidenceKind(StrEnum):
    EVENT = "event"
    TRANSCRIPT_SPAN = "transcript_span"
    CARD_FIELD = "card_field"
    RULE = "rule"


class Evidence(Contract):
    """Проверяемое основание решения (D-031)."""

    kind: EvidenceKind
    ref: NonEmptyStr = Field(description="Идентификатор события, поля, реплики или правила")
    excerpt: str | None = Field(default=None, description="Цитата из ответа ученика")
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _span(self) -> Evidence:
        if self.kind is EvidenceKind.TRANSCRIPT_SPAN and (
            self.start_ms is None or self.end_ms is None
        ):
            raise ValueError("transcript_span requires start_ms and end_ms")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        return self
