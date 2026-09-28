"""Жизненный цикл черновика и gate публикации C-03 (6.5, D-042).

Черновик → проверка структуры и фактов → утверждён преподавателем → опубликован → архив.
Комментарий преподавателя создаёт исправленный черновик (новая версия). Черновик нельзя
назначить обучаемому. Публикационный gate C-03 не обходится утверждением преподавателя:
он применяется отдельно на шаге `publish()`, после `approve()`.
"""

from __future__ import annotations

from datetime import datetime

from ..contracts.routing import RoutingDecision
from ..contracts.scenario import FieldState, Scenario
from ..contracts.scenario_lifecycle import ScenarioDraftRecord, ScenarioStatus

ROUTING_SERVICES_CRITERION = "routing.services"


class LifecycleError(Exception):
    """Недопустимый переход жизненного цикла черновика."""


class GatePublishError(Exception):
    """Gate C-03: сценарий с пустым/неоднозначным обязательным маршрутом не публикуется."""


def assert_assignable(record: ScenarioDraftRecord) -> Scenario:
    """Черновик нельзя назначить обучаемому (6.5) — только опубликованная версия."""
    if not record.assignable:
        raise LifecycleError(
            f"scenario {record.scenario.scenario.version!r} is {record.status}, not "
            f"{ScenarioStatus.PUBLISHED}: a draft cannot be assigned to an attempt"
        )
    return record.scenario


def validate_structure(scenario: Scenario) -> list[str]:
    """Проверка структуры и фактов черновика (6.5). Пустой список — замечаний нет.

    Контрактные инварианты (уникальность тем, покрытие элементов доклада, отсутствие
    выдуманных `expected_fields`) уже проверены при построении `Scenario`; здесь —
    дополнительные содержательные проверки, которые применимы только к публикуемому черновику.
    """
    issues: list[str] = []
    known_topics = {f.topic for f in scenario.published_facts if f.state is FieldState.KNOWN}
    if "address" not in known_topics and not scenario.diagnostic:
        issues.append("published_facts: address is not known (required for a normal scenario)")
    if not scenario.reference.expected_fields:
        issues.append("reference: expected_fields is empty")
    for role in scenario.interlocutors:
        if role.enabled and not role.report_items:
            issues.append(f"interlocutor {role.role_id!r}: enabled but has no report_items")
    return issues


def check_publish_gate(scenario: Scenario, routing: RoutingDecision) -> None:
    """C-03: сценарий с автоматической оценкой маршрута публикуется только с активным правилом.

    Если `routing.services` уже не входит в применимые критерии (`not_applicable_criteria`) —
    маршрут не «обязательный» для этого сценария (например, ожидаемое решение — «отказать», и
    ни одна служба не вызывается в принципе), gate не применяется независимо от статуса
    правила. Иначе — маршрут обязателен и оценивается автоматически, и правило должно быть
    `active`: карантинное, неоднозначное или отсутствующее правило не публикуется как обычный
    источник маршрута. Диагностический сценарий (`Scenario.diagnostic`) — это единственный
    способ учебно показать саму неопределённость, и он обязан нести ту же явную
    неприменимость; иначе он ничем не отличается от обычного сценария с плохим маршрутом и
    блокируется тем же путём. Вызывается на шаге `publish()`, после утверждения преподавателем
    (`approve()`) — утверждение этот gate не обходит.
    """
    if ROUTING_SERVICES_CRITERION in scenario.reference.not_applicable_criteria:
        return
    if routing.auto_gradable:
        return
    raise GatePublishError(
        f"routing rule {routing.rule_id!r} (status={routing.rule_status}) is not active: a "
        "scenario with an applicable 'routing.services' criterion cannot be published with an "
        "unconfirmed mandatory route (C-03). Mark 'routing.services' not_applicable — and the "
        "scenario diagnostic if the point is to teach detecting the uncertainty — or wait for "
        "an active rule."
    )


def submit_for_review(record: ScenarioDraftRecord) -> ScenarioDraftRecord:
    if record.status is not ScenarioStatus.DRAFT:
        raise LifecycleError(f"only a draft can be submitted for review, got {record.status}")
    issues = validate_structure(record.scenario)
    if issues:
        raise LifecycleError("structure/fact check failed: " + "; ".join(issues))
    return record.model_copy(update={"status": ScenarioStatus.VALIDATED})


def approve(record: ScenarioDraftRecord) -> ScenarioDraftRecord:
    if record.status is not ScenarioStatus.VALIDATED:
        raise LifecycleError(f"only a validated draft can be approved, got {record.status}")
    return record.model_copy(update={"status": ScenarioStatus.APPROVED})


def publish(record: ScenarioDraftRecord, routing: RoutingDecision) -> ScenarioDraftRecord:
    """Утверждение преподавателем не обходит gate C-03 — он проверяется здесь заново."""
    if record.status is not ScenarioStatus.APPROVED:
        raise LifecycleError(f"only an approved draft can be published, got {record.status}")
    check_publish_gate(record.scenario, routing)
    return record.model_copy(update={"status": ScenarioStatus.PUBLISHED})


def archive(record: ScenarioDraftRecord) -> ScenarioDraftRecord:
    if record.status is not ScenarioStatus.PUBLISHED:
        raise LifecycleError(f"only a published scenario can be archived, got {record.status}")
    return record.model_copy(update={"status": ScenarioStatus.ARCHIVED})


def revise(
    record: ScenarioDraftRecord,
    *,
    new_scenario: Scenario,
    teacher_comment: str,
    created_at: datetime,
) -> ScenarioDraftRecord:
    """Комментарий преподавателя порождает исправленный черновик — новую версию (6.5, C-09).

    `new_scenario` должен нести новую версию `scenario.scenario` (проверяемо вызывающим);
    исходная опубликованная/утверждённая запись не изменяется (неизменяемость, C-09).
    """
    if new_scenario.scenario == record.scenario.scenario:
        raise LifecycleError("a revision must bump the scenario version (C-09)")
    return record.model_copy(
        update={
            "scenario": new_scenario,
            "status": ScenarioStatus.DRAFT,
            "revises": record.scenario.scenario,
            "teacher_comment": teacher_comment,
            "created_at": created_at,
        }
    )
