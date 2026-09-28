"""Инвариант 12: результат критерия = статус + значение."""

import pytest
from pydantic import ValidationError

from dds_ai.contracts.common import ModelRef
from dds_ai.contracts.criteria import CriterionResult, CriterionStatus, PartialReason, RuleStatus

from .conftest import RULE, ev

S = CriterionStatus


@pytest.mark.parametrize(
    ("status", "value"),
    [(S.PASSED, 1), (S.PASSED, 0.5), (S.FAILED, 0), (S.NOT_DONE, 0)],
)
def test_verified_statuses_accept_their_values(status, value):
    partial = PartialReason.LATE if value == 0.5 else None
    r = CriterionResult(
        criterion_id="c",
        status=status,
        value=value,
        partial_reason=partial,
        evidence=ev(),
        rule_ref=RULE,
    )
    assert r.verified


@pytest.mark.parametrize(
    ("status", "value"),
    [(S.PASSED, 0), (S.FAILED, 1), (S.FAILED, 0.5), (S.NOT_DONE, 1), (S.PASSED, None)],
)
def test_verified_statuses_reject_wrong_values(status, value):
    with pytest.raises(ValidationError):
        CriterionResult(criterion_id="c", status=status, value=value, evidence=ev(), rule_ref=RULE)


@pytest.mark.parametrize(
    "status", [S.NO_EVIDENCE, S.TECHNICAL_ERROR, S.NOT_CHECKED, S.NOT_APPLICABLE]
)
def test_unverified_statuses_carry_no_value(status):
    assert CriterionResult(criterion_id="c", status=status).value is None
    with pytest.raises(ValidationError):
        CriterionResult(criterion_id="c", status=status, value=0)


def test_half_requires_one_of_two_w01_reasons():
    with pytest.raises(ValidationError):
        CriterionResult(criterion_id="c", status=S.PASSED, value=0.5, evidence=ev(), rule_ref=RULE)
    with pytest.raises(ValidationError):
        CriterionResult(
            criterion_id="c",
            status=S.PASSED,
            value=1,
            partial_reason=PartialReason.INACCURATE_FIELD,
            evidence=ev(),
            rule_ref=RULE,
        )


def test_verified_result_requires_evidence_and_basis():
    with pytest.raises(ValidationError):
        CriterionResult(criterion_id="c", status=S.FAILED, value=0, rule_ref=RULE)
    with pytest.raises(ValidationError):
        CriterionResult(criterion_id="c", status=S.FAILED, value=0, evidence=ev())


def test_model_decision_requires_model_ref():
    with pytest.raises(ValidationError):
        CriterionResult(
            criterion_id="c",
            status=S.PASSED,
            value=1,
            evidence=ev(),
            rule_ref=RULE,
            decided_by="model",
        )
    mref = ModelRef(
        component="semantic_judge", model_name="m", model_version="1", prompt_version="p1"
    )
    r = CriterionResult(
        criterion_id="c",
        status=S.PASSED,
        value=1,
        evidence=ev(),
        model_ref=mref,
        decided_by="model",
    )
    assert r.model_ref.prompt_version == "p1"


@pytest.mark.parametrize("rule_status", [RuleStatus.AMBIGUOUS, RuleStatus.QUARANTINED])
def test_non_active_rule_cannot_auto_grade(rule_status):
    with pytest.raises(ValidationError, match="expert"):
        CriterionResult(
            criterion_id="c",
            status=S.FAILED,
            value=0,
            evidence=ev(),
            rule_ref=RULE,
            rule_status=rule_status,
        )
    # Эксперт может вынести вердикт; без эксперта — только not_checked.
    CriterionResult(
        criterion_id="c",
        status=S.FAILED,
        value=0,
        evidence=ev(),
        rule_status=rule_status,
        decided_by="expert",
    )
    CriterionResult(criterion_id="c", status=S.NOT_CHECKED, rule_ref=RULE, rule_status=rule_status)


def test_active_rule_auto_grades():
    CriterionResult(
        criterion_id="c",
        status=S.FAILED,
        value=0,
        evidence=ev(),
        rule_ref=RULE,
        rule_status=RuleStatus.ACTIVE,
    )


def test_results_are_immutable():
    r = CriterionResult(criterion_id="c", status=S.NOT_CHECKED)
    with pytest.raises(ValidationError):
        r.status = S.PASSED
