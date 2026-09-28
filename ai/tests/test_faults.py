"""C-04: инъекция отказов, журнал сбоев, сбой модели ≠ ошибка ученика (инвариант 4)."""

from uuid import uuid4

import pytest

from dds_ai.contracts.common import ModelRef
from dds_ai.contracts.criteria import CriterionResult, CriterionStatus
from dds_ai.contracts.events import ComponentName, FailureKind
from dds_ai.evaluation import judge_criterion
from dds_ai.faults import DEGRADATION, ComponentFailure, Degradation, FaultInjector, InvalidOutput

from .conftest import RULE, T0, ev


class FakeJudge:
    model_ref = ModelRef(component="semantic_judge", model_name="fake", model_version="0")

    def __init__(self, answer=None, exc: Exception | None = None):
        self.answer = answer
        self.exc = exc

    def judge(self, criterion_id, context):
        if self.exc:
            raise self.exc
        return self.answer or CriterionResult(
            criterion_id=criterion_id,
            status=CriterionStatus.PASSED,
            value=1,
            evidence=ev(),
            rule_ref=RULE,
        )


def test_every_component_has_a_degradation():
    assert set(DEGRADATION) == set(ComponentName)
    assert DEGRADATION[ComponentName.SEMANTIC_JUDGE] is Degradation.NOT_CHECKED
    assert DEGRADATION[ComponentName.LLM] is Degradation.SCRIPTED_FALLBACK


@pytest.mark.parametrize("component", list(ComponentName))
@pytest.mark.parametrize("kind", list(FailureKind))
def test_injected_fault_raises_and_is_logged(component, kind):
    inj = FaultInjector(clock=lambda: T0)
    inj.inject(component, kind)
    attempt = uuid4()
    with pytest.raises(ComponentFailure) as info:
        inj.call(component, lambda: "ok", attempt_id=attempt, affected_evidence=("u1",))
    assert (info.value.component, info.value.kind) == (component, kind)
    [rec] = inj.log
    assert (rec.component, rec.kind, rec.attempt_id, rec.affected_evidence) == (
        component,
        kind,
        attempt,
        ("u1",),
    )


def test_other_components_unaffected_and_clear_restores():
    inj = FaultInjector()
    inj.inject(ComponentName.STT, FailureKind.ERROR)
    assert inj.call(ComponentName.TTS, lambda: 1) == 1
    inj.clear(ComponentName.STT)
    assert inj.call(ComponentName.STT, lambda: 2) == 2
    assert inj.log == []


def test_real_timeout_and_crash_are_classified():
    inj = FaultInjector()

    def timeout():
        raise TimeoutError

    def crash():
        raise RuntimeError("boom")

    for fn, kind in ((timeout, FailureKind.TIMEOUT), (crash, FailureKind.ERROR)):
        with pytest.raises(ComponentFailure) as info:
            inj.call(ComponentName.LLM, fn)
        assert info.value.kind is kind


@pytest.mark.parametrize("kind", list(FailureKind))
def test_judge_failure_gives_not_checked_not_zero(kind):
    inj = FaultInjector()
    inj.inject(ComponentName.SEMANTIC_JUDGE, kind)
    r = judge_criterion(FakeJudge(), inj, "voice.facts_transferred")
    assert r.status is CriterionStatus.NOT_CHECKED
    assert r.value is None
    assert inj.log[0].kind is kind


def test_invalid_json_from_judge_gives_not_checked():
    inj = FaultInjector()
    bad = {"criterion_id": "c", "status": "failed", "value": 0}  # без доказательства
    r = judge_criterion(FakeJudge(answer=bad), inj, "c")
    assert r.status is CriterionStatus.NOT_CHECKED
    assert inj.log[0].kind is FailureKind.INVALID_OUTPUT


def test_judge_answering_other_criterion_is_invalid():
    inj = FaultInjector()
    r = judge_criterion(FakeJudge(), inj, "c")
    assert r.status is CriterionStatus.PASSED
    other = CriterionResult(criterion_id="x", status=CriterionStatus.NOT_CHECKED)
    r = judge_criterion(FakeJudge(answer=other), inj, "c")
    assert r.status is CriterionStatus.NOT_CHECKED
    assert inj.log[-1].kind is FailureKind.INVALID_OUTPUT


def test_plain_value_error_is_runtime_error_not_invalid_output():
    inj = FaultInjector()

    def boom():
        raise ValueError("prompt longer than context")

    def bad_output():
        raise InvalidOutput("schema")

    for fn, kind in ((boom, FailureKind.ERROR), (bad_output, FailureKind.INVALID_OUTPUT)):
        with pytest.raises(ComponentFailure) as info:
            inj.call(ComponentName.LLM, fn)
        assert info.value.kind is kind
