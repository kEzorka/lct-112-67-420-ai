"""Снимок версий попытки, итог расчёта и версия оценки (D-003, D-038, C-06, C-08).

Черновики: хранение и расчёт — у бэкенда; ИИ-контур поставляет CriterionResult и версии моделей.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from .common import Contract, ModelRef, NonEmptyStr, VersionRef
from .criteria import CriterionResult


class AttemptVersionSnapshot(Contract):
    """Фиксируется 1:1 при старте попытки."""

    attempt_id: UUID
    scenario: VersionRef
    reference: VersionRef
    card_schema: VersionRef
    classifier: VersionRef
    routing_rules: VersionRef
    rubric: VersionRef
    time_policy: VersionRef
    difficulty_config: VersionRef
    models: tuple[ModelRef, ...] = ()
    created_at: AwareDatetime


class Verdict(StrEnum):
    PASSED = "passed"
    NOT_PASSED = "not_passed"
    PROVISIONAL = "provisional"
    NOT_SCORED = "not_scored"


class ScoreSummary(Contract):
    rubric_version: NonEmptyStr
    verdict: Verdict
    total: float | None = Field(default=None, ge=0, le=100)
    lower: float | None = Field(default=None, ge=0, le=100)
    upper: float | None = Field(default=None, ge=0, le=100)
    effective_weights: dict[NonEmptyStr, float] = Field(default_factory=dict)
    excluded_groups: tuple[NonEmptyStr, ...] = ()
    critical_failures: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> ScoreSummary:
        v = self.verdict
        if v is Verdict.NOT_SCORED and (self.total, self.lower, self.upper) != (None, None, None):
            raise ValueError("not_scored carries no numbers")
        if v is Verdict.PROVISIONAL and (
            self.total is not None or self.lower is None or self.upper is None
        ):
            raise ValueError("provisional carries a range, not a total")
        if v in (Verdict.PASSED, Verdict.NOT_PASSED) and self.total is None:
            raise ValueError("final verdict requires total")
        return self

    @property
    def valid_score(self) -> bool:
        """Только такой результат обновляет профиль и рейтинг (C-07, инвариант 13)."""
        return self.verdict in (Verdict.PASSED, Verdict.NOT_PASSED)


class ScoreVersion(Contract):
    """Append-only: экспертная поправка создаёт версию N+1."""

    score_version_id: UUID
    attempt_id: UUID
    parent_score_version_id: UUID | None = None
    author: NonEmptyStr
    reason: str | None = None
    created_at: AwareDatetime
    criterion_results: tuple[CriterionResult, ...]
    summary: ScoreSummary

    @model_validator(mode="after")
    def _correction(self) -> ScoreVersion:
        if self.parent_score_version_id is not None and not self.reason:
            raise ValueError("a correction requires a reason")
        return self
