"""Матрица C-06 + W-01 на эталонном расчёте балла."""

import pytest

from dds_ai.contracts.criteria import CriterionStatus, PartialReason
from dds_ai.contracts.rubric import Rubric
from dds_ai.contracts.scoring import Verdict
from dds_ai.mocks.scoring import score

from .conftest import all_results, result

S = CriterionStatus


def test_all_passed(rubric):
    s = score(rubric, all_results(rubric, S.PASSED))
    assert (s.verdict, s.total) == (Verdict.PASSED, 100)
    assert s.valid_score


def test_all_failed_is_not_passed_with_criticals(rubric):
    s = score(rubric, all_results(rubric, S.FAILED))
    assert (s.verdict, s.total) == (Verdict.NOT_PASSED, 0)
    assert s.critical_failures == ("card.address", "routing.services", "time.processing")


def test_one_not_checked_gives_provisional_range(rubric):
    s = score(
        rubric,
        all_results(
            rubric,
            S.PASSED,
            **{"voice.facts_transferred": result("voice.facts_transferred", S.NOT_CHECKED)},
        ),
    )
    assert s.verdict is Verdict.PROVISIONAL
    assert s.total is None
    # voice = 25, facts_transferred = 2/4 группы → 12.5
    assert (s.lower, s.upper) == (87.5, 100)
    assert not s.valid_score


def test_missed_call_is_not_done_and_final(rubric):
    """Пропущенный звонок при исправной системе: not_done, 0, итог НЕ предварительный."""
    s = score(
        rubric,
        all_results(rubric, S.PASSED, **{"voice.call_made": result("voice.call_made", S.NOT_DONE)}),
    )
    assert s.verdict is Verdict.PASSED
    assert s.total == pytest.approx(100 - 25 / 4)


@pytest.mark.parametrize("status", [S.NO_EVIDENCE, S.TECHNICAL_ERROR])
def test_lost_audio_is_provisional(rubric, status):
    s = score(
        rubric,
        all_results(
            rubric, S.PASSED, **{"voice.ack_received": result("voice.ack_received", status)}
        ),
    )
    assert s.verdict is Verdict.PROVISIONAL


def test_not_applicable_group_is_excluded_and_rest_renormalised(rubric):
    overrides = {
        c.criterion_id: result(c.criterion_id, S.NOT_APPLICABLE) for c in rubric.groups[2].criteria
    }
    overrides["card.address"] = result("card.address", S.FAILED)
    s = score(rubric, all_results(rubric, S.PASSED, **overrides))
    assert s.excluded_groups == ("voice",)
    assert sum(s.effective_weights.values()) == pytest.approx(100)
    # card = 20/75*100; address = 2/4 карточки
    assert s.total == pytest.approx(100 - 20 / 75 * 100 / 2)


def test_no_applicable_criteria_is_not_scored(rubric):
    s = score(rubric, all_results(rubric, S.NOT_APPLICABLE))
    assert s.verdict is Verdict.NOT_SCORED
    assert (s.total, s.lower, s.upper) == (None, None, None)


def test_late_open_gets_half_and_still_passes(rubric):
    late = result("time.open", S.PASSED, 0.5, PartialReason.LATE)
    s = score(rubric, all_results(rubric, S.PASSED, **{"time.open": late}))
    assert s.verdict is Verdict.PASSED
    assert s.total == pytest.approx(100 - 20 / 3 / 2)


def test_processing_over_180_fails_regardless_of_total(rubric):
    s = score(
        rubric,
        all_results(rubric, S.PASSED, **{"time.processing": result("time.processing", S.FAILED)}),
    )
    assert s.total > rubric.pass_threshold
    assert s.verdict is Verdict.NOT_PASSED
    assert s.critical_failures == ("time.processing",)


def test_unverified_critical_is_not_a_critical_failure(rubric):
    s = score(
        rubric,
        all_results(
            rubric, S.PASSED, **{"routing.services": result("routing.services", S.TECHNICAL_ERROR)}
        ),
    )
    assert s.verdict is Verdict.PROVISIONAL
    assert s.critical_failures == ()


def test_threshold_75_boundary(rubric):
    # Провал всей группы «Голос» (25) без критических → ровно 75 → сдано.
    overrides = {
        c.criterion_id: result(c.criterion_id, S.FAILED) for c in rubric.groups[2].criteria
    }
    s = score(rubric, all_results(rubric, S.PASSED, **overrides))
    assert s.total == pytest.approx(75)
    assert s.verdict is Verdict.PASSED


def test_adding_criterion_does_not_change_group_weight(rubric):
    data = rubric.model_dump(mode="json")
    data["groups"][0]["criteria"].append(
        {"criterion_id": "card.extra", "title": "x", "weight": 5, "critical": False}
    )
    bigger = Rubric.model_validate(data)
    s = score(bigger, all_results(bigger, S.PASSED))
    card = sum(v for k, v in s.effective_weights.items() if k.startswith("card."))
    assert card == pytest.approx(20)


def test_every_result_must_match_rubric(rubric):
    with pytest.raises(ValueError, match="no result"):
        score(rubric, all_results(rubric, S.PASSED)[1:])
