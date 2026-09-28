"""Матрица покрытия синтетического набора (6.5): решение × профиль ДДС × состояние вызова.

Сверяет `docs/ai/coverage-matrix.md` с фактическим набором `ai/data/synthetic/*.json`: если
кто-то добавит или уберёт сценарий, не обновив документ, этот тест укажет на расхождение.
"""

from __future__ import annotations

from dds_ai.contracts.scenario import Provenance
from dds_ai.cycle import load_synthetic
from dds_ai.scenarios.coverage import (
    CoverageCell,
    coverage_cells,
    decisions_covered,
    profiles_covered,
)

# Соответствует таблице 1 в docs/ai/coverage-matrix.md.
EXPECTED_CELLS = {
    CoverageCell("respond", "dds-center", "connected"),
    CoverageCell("refuse", "dds-center", "dropped"),
    CoverageCell("redirect", "dds-center", "busy"),
    CoverageCell("redirect", "dds-center", "no_answer"),
    CoverageCell("respond", "dds-north", "connected"),
    CoverageCell("redirect", "dds-north", "no_answer"),
    CoverageCell("diagnostic", "dds-north", "connected"),
    CoverageCell("respond", "dds-river", "connected"),
    CoverageCell("refuse", "dds-river", "dropped"),
    CoverageCell("redirect", "dds-river", "busy"),
}


def test_coverage_matches_the_documented_matrix():
    scenarios = load_synthetic().values()
    assert coverage_cells(scenarios) == EXPECTED_CELLS


def test_all_three_dispatcher_decisions_are_covered_plus_diagnostic():
    scenarios = load_synthetic().values()
    assert decisions_covered(scenarios) == {"respond", "refuse", "redirect", "diagnostic"}


def test_three_dds_profiles_are_covered():
    scenarios = load_synthetic().values()
    assert profiles_covered(scenarios) == {"dds-center", "dds-north", "dds-river"}


def test_exactly_one_diagnostic_scenario_documented_as_a_gap_elsewhere():
    scenarios = load_synthetic().values()
    diagnostic = [s for s in scenarios if s.diagnostic]
    assert len(diagnostic) == 1
    assert diagnostic[0].scenario.name == "syn-006-unclassified-diagnostic"


def test_set_size_is_within_the_required_range():
    assert 8 <= len(load_synthetic()) <= 12


def test_every_synthetic_scenario_is_flagged_synthetic_not_a_real_ticket():
    for scenario in load_synthetic().values():
        assert scenario.provenance is Provenance.SYNTHETIC
        assert scenario.source_ref is None
        assert scenario.title.startswith("СИНТЕТИКА")
