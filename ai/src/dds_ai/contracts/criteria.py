"""Результат критерия = статус + значение (инвариант 12, C-06, W-01)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .common import Contract, Evidence, ModelRef, NonEmptyStr, VersionRef


class CriterionStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_DONE = "not_done"
    NO_EVIDENCE = "no_evidence"
    TECHNICAL_ERROR = "technical_error"
    NOT_APPLICABLE = "not_applicable"
    NOT_CHECKED = "not_checked"


VERIFIED = frozenset({CriterionStatus.PASSED, CriterionStatus.FAILED, CriterionStatus.NOT_DONE})
UNVERIFIED = frozenset(
    {CriterionStatus.NO_EVIDENCE, CriterionStatus.TECHNICAL_ERROR, CriterionStatus.NOT_CHECKED}
)


class PartialReason(StrEnum):
    """Единственные два основания для 0,5 по W-01."""

    INACCURATE_FIELD = "inaccurate_field"
    LATE = "late"


class RuleStatus(StrEnum):
    ACTIVE = "active"
    AMBIGUOUS = "ambiguous"
    QUARANTINED = "quarantined"


Value = Literal[0, 0.5, 1]


class CriterionResult(Contract):
    """Результат одного критерия.

    - Значение есть только у проверенных статусов: passed → 1 или 0,5; failed и not_done → 0.
    - 0,5 требует основания partial_reason (W-01).
    - Проверенный результат требует доказательства и ссылки на правило или модель (D-031).
    - По правилу со статусом ambiguous/quarantined авто-вердикт запрещён: только эксперт.
    """

    criterion_id: NonEmptyStr
    status: CriterionStatus
    value: Value | None = None
    partial_reason: PartialReason | None = None
    evidence: tuple[Evidence, ...] = ()
    rule_ref: VersionRef | None = None
    rule_status: RuleStatus | None = None
    model_ref: ModelRef | None = None
    decided_by: Literal["rule", "model", "expert"] = "rule"
    explanation: str | None = Field(default=None, description="Понятное обоснование для ученика")

    @model_validator(mode="after")
    def _invariant_12(self) -> CriterionResult:
        s = self.status
        if s in VERIFIED:
            if self.value is None:
                raise ValueError(f"{s} requires value")
            allowed = {CriterionStatus.PASSED: {1, 0.5}}.get(s, {0})
            if self.value not in allowed:
                raise ValueError(f"{s} allows value in {sorted(allowed)}, got {self.value}")
            if not self.evidence:
                raise ValueError(f"{s} requires evidence")
            if self.rule_ref is None and self.model_ref is None and self.decided_by != "expert":
                raise ValueError(f"{s} requires rule_ref or model_ref")
            if (
                self.decided_by != "expert"
                and self.rule_status is not None
                and self.rule_status is not RuleStatus.ACTIVE
            ):
                raise ValueError("only an active rule may auto-pass/fail; send to expert")
        elif self.value is not None:
            raise ValueError(f"{s} must not carry a value")

        if (self.value == 0.5) != (self.partial_reason is not None):
            raise ValueError("partial_reason is required exactly when value is 0.5")
        if self.decided_by == "model" and self.model_ref is None:
            raise ValueError("model decision requires model_ref")
        return self

    @property
    def verified(self) -> bool:
        return self.status in VERIFIED

    @property
    def unverified(self) -> bool:
        return self.status in UNVERIFIED
