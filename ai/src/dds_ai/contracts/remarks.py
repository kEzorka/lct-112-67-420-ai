"""Замечания (D-037)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from .common import Contract, Evidence, ModelRef, NonEmptyStr


class RemarkType(StrEnum):
    TIME = "time"
    FACT_FIELD = "fact_field"
    ACTION_ROUTE = "action_route"
    COMMUNICATION = "communication"
    SEQUENCE = "sequence"
    GRAMMAR = "grammar"
    SOURCE_UNCERTAINTY = "source_uncertainty"
    TECHNICAL_FAULT = "technical_fault"


NOT_STUDENT_ERRORS = frozenset({RemarkType.SOURCE_UNCERTAINTY, RemarkType.TECHNICAL_FAULT})


class Severity(StrEnum):
    INFO = "info"
    ERROR = "error"
    MAJOR_ERROR = "major_error"


class Remark(Contract):
    remark_id: NonEmptyStr
    attempt_id: NonEmptyStr
    type: RemarkType
    severity: Severity
    criterion_id: str | None = None
    text: NonEmptyStr
    suggestion: str | None = Field(
        default=None, description="Для грамматики: исправление, не применяется"
    )
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    owner: NonEmptyStr = Field(description="Кто проверяет: rule / model / expert")
    model_ref: ModelRef | None = None

    @property
    def counts_as_error(self) -> bool:
        """Сбой техники и неопределённость эталона не увеличивают счётчик ошибок ученика."""
        return self.type not in NOT_STUDENT_ERRORS and self.severity is not Severity.INFO


class ErrorExplanation(Contract):
    """Объяснение ошибки после завершения попытки (Q&A 8, D-041, D-045): ошибка → доказательство
    → упражнение для исправления. Строится только из замечаний-ошибок ученика
    (`Remark.counts_as_error`), не из технических сбоев или неопределённости источника."""

    remark_id: NonEmptyStr
    error: NonEmptyStr
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    exercise: NonEmptyStr
