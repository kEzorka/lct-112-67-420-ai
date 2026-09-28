"""Вектор сложности (D-043, 6.8): осложняющие факторы, шкала 0/1/2, версия весов.

ИИ предлагает вектор и объяснение по каждому фактору; публикует и утверждает границы
уровней — преподаватель (эта фаза не в объёме ИИ-контура, см. `docs/ai/progress/m6.md`).
Сложность не меняет нормативы 30/180 (C-01) — эти контракты не ссылаются на время попытки.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .common import Contract, ModelRef, NonEmptyStr, VersionRef

FactorLevel = Literal[0, 1, 2]


class DifficultyFactor(StrEnum):
    """Начальные факторы (D-043); начальные веса равны — см. `ai/config/difficulty.json`."""

    AMBIGUITY = "ambiguity"  # неоднозначность исходных сведений
    CIRCUMSTANCES = "circumstances"  # число существенных обстоятельств
    SERVICES = "services"  # число вовлечённых служб
    ADDITIONS = "additions"  # объём необходимых дополнений
    SPEECH = "speech"  # речевые осложнения разговора


ALL_FACTORS: tuple[DifficultyFactor, ...] = tuple(DifficultyFactor)


def _require_all_factors(keys: set) -> None:
    missing = set(DifficultyFactor) - keys
    extra = keys - set(DifficultyFactor)
    if missing or extra:
        raise ValueError(
            f"vector must cover exactly the 5 factors; missing={missing} extra={extra}"
        )


class DifficultyWeights(Contract):
    """Версионируемые веса факторов (D-043): начальные — равные, публикует преподаватель."""

    weights_version: NonEmptyStr
    weights: dict[DifficultyFactor, float] = Field(description="Вес каждого фактора, ≥ 0")

    @model_validator(mode="after")
    def _shape(self) -> DifficultyWeights:
        _require_all_factors(set(self.weights))
        negative = {f for f, w in self.weights.items() if w < 0}
        if negative:
            raise ValueError(f"weights must be non-negative: {sorted(negative)}")
        return self


class DifficultyVector(Contract):
    """Утверждённый вектор факторов для сценария (0/1/2 по каждому)."""

    values: dict[DifficultyFactor, FactorLevel]

    @model_validator(mode="after")
    def _shape(self) -> DifficultyVector:
        _require_all_factors(set(self.values))
        return self


class FactorContribution(Contract):
    factor: DifficultyFactor
    level: FactorLevel
    weight: float = Field(ge=0)
    contribution: float = Field(ge=0, description="level * weight — вклад фактора в сумму")


class DifficultyScore(Contract):
    """Результат расчёта: сумма видна по фактору (приёмка D-043)."""

    scenario: VersionRef
    weights_version: NonEmptyStr
    contributions: tuple[FactorContribution, ...] = Field(min_length=5, max_length=5)
    total: float = Field(ge=0)

    @model_validator(mode="after")
    def _shape(self) -> DifficultyScore:
        _require_all_factors({c.factor for c in self.contributions})
        total = round(sum(c.contribution for c in self.contributions), 9)
        if abs(total - round(self.total, 9)) > 1e-6:
            raise ValueError("total must equal the sum of per-factor contributions")
        return self


class FactorProposal(Contract):
    """Предложение ИИ по одному фактору — с объяснением (D-043: «ИИ предлагает с объяснением»)."""

    factor: DifficultyFactor
    level: FactorLevel
    rationale: NonEmptyStr


class DifficultyProposal(Contract):
    """Предложение вектора целиком; публикует преподаватель (это действие — вне ИИ-контура)."""

    scenario: VersionRef
    factors: tuple[FactorProposal, ...] = Field(min_length=5, max_length=5)
    model_ref: ModelRef | None = Field(
        default=None, description="None — детерминированное правило, не модель"
    )

    @model_validator(mode="after")
    def _shape(self) -> DifficultyProposal:
        _require_all_factors({f.factor for f in self.factors})
        return self

    @property
    def vector(self) -> DifficultyVector:
        return DifficultyVector(values={f.factor: f.level for f in self.factors})
