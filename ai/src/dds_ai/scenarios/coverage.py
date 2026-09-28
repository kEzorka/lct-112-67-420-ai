"""Матрица покрытия синтетического набора сценариев (6.5): решение диспетчера × профиль ДДС ×
состояние вызова (связь/сбой). `docs/ai/coverage-matrix.md` — ручной документ для
преподавателя; этот модуль даёт проверяемые данные, которыми тест сверяет документ с
фактическим набором `ai/data/synthetic/*.json`, чтобы текст не разошёлся с кодом.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..contracts.scenario import Scenario


@dataclass(frozen=True)
class CoverageCell:
    decision: str  # respond / refuse / redirect / diagnostic
    dds_profile: str
    call_condition: str  # connected / busy / no_answer / dropped


def _call_conditions(scenario: Scenario) -> set[str]:
    conditions: set[str] = set()
    for role in scenario.interlocutors:
        if not role.enabled:
            continue
        if role.drop_after_turns is not None:
            conditions.add("dropped")
        conditions.update(outcome.value for outcome in role.dial_outcomes)
    return conditions or {"connected"}


def coverage_cells(scenarios: Iterable[Scenario]) -> set[CoverageCell]:
    """Одна ячейка на (решение, профиль, состояние вызова), встреченные в наборе."""
    cells: set[CoverageCell] = set()
    for s in scenarios:
        decision = "diagnostic" if s.diagnostic else s.reference.expected_decision.value
        for condition in _call_conditions(s):
            cells.add(CoverageCell(decision, s.dds_profile, condition))
    return cells


def decisions_covered(scenarios: Iterable[Scenario]) -> set[str]:
    return {c.decision for c in coverage_cells(scenarios)}


def profiles_covered(scenarios: Iterable[Scenario]) -> set[str]:
    return {c.dds_profile for c in coverage_cells(scenarios)}
