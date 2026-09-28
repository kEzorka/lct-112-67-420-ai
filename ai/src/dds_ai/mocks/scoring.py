"""Эталонная реализация расчёта балла по C-06 + W-01.

Настоящий модуль оценивания — у бэкенда. Этот нужен, чтобы тестами доказать, что выход
ИИ-контура (CriterionResult) корректно ложится в расчёт, и чтобы свериться с реализацией бэкенда.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..contracts.criteria import CriterionResult, CriterionStatus
from ..contracts.rubric import Rubric
from ..contracts.scoring import ScoreSummary, Verdict


def score(rubric: Rubric, results: Iterable[CriterionResult]) -> ScoreSummary:
    by_id = {r.criterion_id: r for r in results}
    missing = {c.criterion_id for g in rubric.groups for c in g.criteria} - set(by_id)
    if missing:
        raise ValueError(f"no result for criteria: {sorted(missing)}")
    unknown = set(by_id) - {c.criterion_id for g in rubric.groups for c in g.criteria}
    if unknown:
        raise ValueError(f"results for criteria outside rubric: {sorted(unknown)}")

    # Применимость групп и нормировка (C-06).
    applicable: dict[str, list] = {}
    for g in rubric.groups:
        crits = [
            c
            for c in g.criteria
            if by_id[c.criterion_id].status is not CriterionStatus.NOT_APPLICABLE
        ]
        if crits:
            applicable[g.group_id] = crits
    excluded = tuple(g.group_id for g in rubric.groups if g.group_id not in applicable)

    if not applicable:
        return ScoreSummary(
            rubric_version=rubric.rubric_version,
            verdict=Verdict.NOT_SCORED,
            excluded_groups=excluded,
        )

    group_total = sum(g.weight for g in rubric.groups if g.group_id in applicable)
    eff: dict[str, float] = {}
    for g in rubric.groups:
        crits = applicable.get(g.group_id)
        if not crits:
            continue
        g_weight = g.weight * 100 / group_total
        c_total = sum(c.weight for c in crits)
        for c in crits:
            eff[c.criterion_id] = g_weight * c.weight / c_total

    lower = upper = 0.0
    unverified = False
    for cid, w in eff.items():
        r = by_id[cid]
        if r.unverified:
            unverified = True
            upper += w
        else:
            lower += w * float(r.value)
            upper += w * float(r.value)

    critical_ids = {c.criterion_id for g in rubric.groups for c in g.criteria if c.critical}
    # Критическая ошибка — только проверенный факт (W-01).
    critical_failures = tuple(
        sorted(
            cid
            for cid in critical_ids & eff.keys()
            if by_id[cid].verified and by_id[cid].value == 0
        )
    )

    common = {
        "rubric_version": rubric.rubric_version,
        "effective_weights": {k: round(v, 6) for k, v in eff.items()},
        "excluded_groups": excluded,
        "critical_failures": critical_failures,
    }
    lower, upper = round(lower, 6), round(upper, 6)
    if unverified:
        return ScoreSummary(verdict=Verdict.PROVISIONAL, lower=lower, upper=upper, **common)
    passed = lower >= rubric.pass_threshold and not critical_failures
    return ScoreSummary(
        verdict=Verdict.PASSED if passed else Verdict.NOT_PASSED, total=lower, **common
    )
