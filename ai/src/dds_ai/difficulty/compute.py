"""Расчёт сложности: взвешенная сумма по вектору факторов (D-043).

Детерминировано: тот же вектор и версия весов дают тот же результат. Веса неотрицательны
(`DifficultyWeights`), поэтому добавление осложнения (рост уровня фактора) никогда не
уменьшает сумму — структурно, не только по тесту.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts.common import VersionRef
from ..contracts.difficulty import (
    ALL_FACTORS,
    DifficultyFactor,
    DifficultyScore,
    DifficultyVector,
    DifficultyWeights,
    FactorContribution,
)

CONFIG = Path(__file__).resolve().parents[3] / "config"
DEFAULT_WEIGHTS = CONFIG / "difficulty.json"


def load_weights(path: Path = DEFAULT_WEIGHTS) -> DifficultyWeights:
    data = json.loads(path.read_text("utf-8"))
    return DifficultyWeights(weights_version=data["weights_version"], weights=data["weights"])


def score(
    vector: DifficultyVector, weights: DifficultyWeights, *, scenario: VersionRef
) -> DifficultyScore:
    contributions = tuple(
        FactorContribution(
            factor=factor,
            level=vector.values[factor],
            weight=weights.weights[factor],
            contribution=round(vector.values[factor] * weights.weights[factor], 9),
        )
        for factor in ALL_FACTORS
    )
    total = round(sum(c.contribution for c in contributions), 9)
    return DifficultyScore(
        scenario=scenario,
        weights_version=weights.weights_version,
        contributions=contributions,
        total=total,
    )


def contribution_of(result: DifficultyScore, factor: DifficultyFactor) -> FactorContribution:
    return next(c for c in result.contributions if c.factor is factor)
