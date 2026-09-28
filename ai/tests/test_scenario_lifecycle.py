"""Жизненный цикл черновика и gate C-03 (6.5, D-042): черновик нельзя назначить; обычный
сценарий с пустым/неоднозначным маршрутом не публикуется; диагностический публикуется только
с явной неприменимостью; утверждение преподавателя gate не обходит."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dds_ai.contracts.card import FieldState
from dds_ai.contracts.common import VersionRef
from dds_ai.contracts.criteria import RuleStatus
from dds_ai.contracts.dialogue import Interlocutor, SupervisorAction
from dds_ai.contracts.events import DispatcherDecision
from dds_ai.contracts.routing import RoutingDecision
from dds_ai.contracts.scenario import (
    InterlocutorBrief,
    Provenance,
    Scenario,
    ScenarioFact,
    ScenarioReference,
)
from dds_ai.contracts.scenario_lifecycle import (
    GenerationInputs,
    GenerationMeta,
    ScenarioDraftRecord,
    ScenarioStatus,
)
from dds_ai.scenarios.lifecycle import (
    GatePublishError,
    LifecycleError,
    approve,
    archive,
    assert_assignable,
    check_publish_gate,
    publish,
    revise,
    submit_for_review,
    validate_structure,
)

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
RULES = VersionRef(name="routing_rules", version="mock-1")


def _brief(report_items=True) -> InterlocutorBrief:
    from dds_ai.contracts.scenario import CheckableItem, ReportElement

    items = (
        (
            CheckableItem(item_id="e", element=ReportElement.ESSENCE, patterns=("пожар",)),
            CheckableItem(item_id="l", element=ReportElement.LOCATION, patterns=("тест",)),
            CheckableItem(item_id="c", element=ReportElement.CIRCUMSTANCES, patterns=("дым",)),
            CheckableItem(item_id="d", element=ReportElement.DDS_DECISION, patterns=("реагир",)),
            CheckableItem(item_id="r", element=ReportElement.REQUESTED_ACTION, patterns=("прошу",)),
        )
        if report_items
        else ()
    )
    return InterlocutorBrief(
        interlocutor=Interlocutor(role_id="supervisor", allowed_actions=(SupervisorAction.LISTEN,)),
        report_items=items,
    )


def _scenario(
    *,
    version: str = "1",
    incident_type: str = "fire_residential",
    diagnostic: bool = False,
    not_applicable: tuple[str, ...] = (),
    report_items: bool = True,
) -> Scenario:
    return Scenario(
        scenario=VersionRef(name="gen-lifecycle-test", version=version),
        title="СИНТЕТИКА: тест жизненного цикла",
        provenance=Provenance.SYNTHETIC,
        incident_type=incident_type,
        diagnostic=diagnostic,
        published_facts=(
            ScenarioFact(
                fact_id="card.address",
                topic="address",
                label="Адрес",
                state=FieldState.KNOWN,
                value="г. Учебный, ул. Тестовая, д. 1",
            ),
        ),
        interlocutors=(_brief(report_items=report_items),),
        reference=ScenarioReference(
            reference=VersionRef(name="gen-lifecycle-test.ref", version="1"),
            expected_decision=DispatcherDecision.RESPOND,
            expected_fields={"address": "г. Учебный, ул. Тестовая, д. 1"},
            not_applicable_criteria=not_applicable,
        ),
    )


def _record(
    scenario: Scenario, status: ScenarioStatus = ScenarioStatus.DRAFT
) -> ScenarioDraftRecord:
    return ScenarioDraftRecord(
        scenario=scenario,
        status=status,
        generation=GenerationMeta(
            inputs=GenerationInputs(
                incident_type=scenario.incident_type,
                location="г. Учебный, ул. Тестовая, д. 1",
                difficulty="medium",
                topic="дым",
            ),
            template_version="draft-template-1",
        ),
        created_at=T0,
    )


def _active_route() -> RoutingDecision:
    return RoutingDecision(
        routing_rules=RULES,
        rule_id="r1",
        rule_status=RuleStatus.ACTIVE,
        ekp_code="1",
        main_service="fire_service",
        services=("fire_service",),
    )


def _no_rule_route() -> RoutingDecision:
    return RoutingDecision(routing_rules=RULES, rule_id=None, rule_status=None)


def _quarantined_route() -> RoutingDecision:
    return RoutingDecision(routing_rules=RULES, rule_id="r2", rule_status=RuleStatus.QUARANTINED)


# --- нельзя назначить черновик --------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        ScenarioStatus.DRAFT,
        ScenarioStatus.VALIDATED,
        ScenarioStatus.APPROVED,
        ScenarioStatus.ARCHIVED,
    ],
)
def test_draft_cannot_be_assigned(status: ScenarioStatus):
    record = _record(_scenario(), status=status)
    with pytest.raises(LifecycleError):
        assert_assignable(record)


def test_published_scenario_can_be_assigned():
    record = _record(_scenario(), status=ScenarioStatus.PUBLISHED)
    assert assert_assignable(record) == record.scenario


# --- переходы жизненного цикла ---------------------------------------------------------------


def test_full_lifecycle_happy_path():
    record = _record(_scenario())
    record = submit_for_review(record)
    assert record.status is ScenarioStatus.VALIDATED
    record = approve(record)
    assert record.status is ScenarioStatus.APPROVED
    record = publish(record, _active_route())
    assert record.status is ScenarioStatus.PUBLISHED
    record = archive(record)
    assert record.status is ScenarioStatus.ARCHIVED


def test_cannot_skip_review_or_approval():
    record = _record(_scenario())
    with pytest.raises(LifecycleError):
        approve(record)
    with pytest.raises(LifecycleError):
        publish(record, _active_route())


def test_submit_for_review_rejects_structurally_broken_draft():
    broken = _scenario(report_items=False)
    record = _record(broken)
    with pytest.raises(LifecycleError):
        submit_for_review(record)
    assert validate_structure(broken)  # непустой список замечаний


def test_teacher_comment_creates_a_new_corrected_draft():
    record = submit_for_review(_record(_scenario(version="1")))
    record = approve(record)
    fixed = _scenario(version="2")
    revised = revise(record, new_scenario=fixed, teacher_comment="уточните адрес", created_at=T0)
    assert revised.status is ScenarioStatus.DRAFT
    assert revised.revises == VersionRef(name="gen-lifecycle-test", version="1")
    assert revised.teacher_comment == "уточните адрес"
    # исходная запись не изменилась (C-09)
    assert record.status is ScenarioStatus.APPROVED
    assert record.scenario.scenario.version == "1"


def test_revise_requires_a_new_scenario_version():
    record = _record(_scenario(version="1"))
    with pytest.raises(LifecycleError):
        revise(
            record,
            new_scenario=_scenario(version="1"),
            teacher_comment="x",
            created_at=T0,
        )


# --- gate C-03 --------------------------------------------------------------------------------


def test_non_diagnostic_scenario_with_route_already_inapplicable_publishes_regardless_of_rule():
    """`routing.services` может быть неприменим не из-за неопределённости маршрута, а по
    педагогической причине (решение «отказать» — служба не вызывается вовсе, как в
    `syn-003-repeat-refuse.json`). Такой сценарий не обязан быть диагностическим: маршрут для
    него не «обязательный» критерий, gate C-03 к нему не применяется независимо от статуса
    правила."""
    scenario = _scenario(diagnostic=False, not_applicable=("routing.services",))
    record = approve(submit_for_review(_record(scenario)))
    published = publish(record, _quarantined_route())
    assert published.status is ScenarioStatus.PUBLISHED


def test_normal_scenario_with_no_rule_cannot_be_published():
    record = approve(submit_for_review(_record(_scenario(diagnostic=False))))
    with pytest.raises(GatePublishError):
        publish(record, _no_rule_route())


def test_normal_scenario_with_quarantined_rule_cannot_be_published():
    record = approve(submit_for_review(_record(_scenario(diagnostic=False))))
    with pytest.raises(GatePublishError):
        publish(record, _quarantined_route())


def test_diagnostic_scenario_without_exemption_still_blocked():
    """Диагностический сценарий без явной неприменимости routing.services тоже не публикуется."""
    scenario = _scenario(diagnostic=True, not_applicable=())
    record = approve(submit_for_review(_record(scenario)))
    with pytest.raises(GatePublishError):
        publish(record, _no_rule_route())


def test_diagnostic_scenario_with_exemption_publishes():
    scenario = _scenario(diagnostic=True, not_applicable=("routing.services",))
    record = approve(submit_for_review(_record(scenario)))
    published = publish(record, _no_rule_route())
    assert published.status is ScenarioStatus.PUBLISHED


def test_normal_scenario_with_active_rule_publishes_normally():
    record = approve(submit_for_review(_record(_scenario())))
    published = publish(record, _active_route())
    assert published.status is ScenarioStatus.PUBLISHED


def test_teacher_approval_does_not_bypass_the_gate():
    """Утверждение преподавателем (approve) проходит; сам gate проверяется отдельно на publish
    и всё равно блокирует сценарий с неподтверждённым маршрутом (C-03)."""
    scenario = _scenario(diagnostic=False)
    record = approve(submit_for_review(_record(scenario)))
    assert record.status is ScenarioStatus.APPROVED  # утверждение прошло
    with pytest.raises(GatePublishError):
        publish(record, _no_rule_route())  # но публикация всё равно блокируется
    check_publish_gate(scenario, _active_route())  # с активным правилом gate не мешает
