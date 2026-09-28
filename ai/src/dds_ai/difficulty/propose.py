"""ИИ предлагает вектор сложности с объяснением по фактору (D-043); публикует преподаватель.

Правило, не LLM (решение команды, как `card.address`/`manual.grammar` в M5): сложность
считается по наблюдаемым свойствам уже опубликованного сценария — эталон не нужно
интерпретировать моделью, чтобы посчитать, сколько в нём фактов, служб и осложнений.
Пороги внутри правила — рабочая конвенция команды, не утверждённая методика; преподаватель
может не согласиться с конкретным уровнем — для этого и нужно `rationale` по каждому фактору.
"""

from __future__ import annotations

from ..contracts.card import FieldState
from ..contracts.dialogue import ScenarioCallState
from ..contracts.difficulty import DifficultyFactor, DifficultyProposal, FactorProposal
from ..contracts.routing import RoutingDecision
from ..contracts.scenario import Scenario

_DEDICATED_TOPICS = frozenset({"address", "incident_type", "services"})


def _level(count: int, *, one: int, two: int) -> int:
    if count >= two:
        return 2
    if count >= one:
        return 1
    return 0


def _ambiguity(scenario: Scenario) -> FactorProposal:
    unresolved = [f for f in scenario.published_facts if f.state is not FieldState.KNOWN]
    level = _level(len(unresolved), one=1, two=2)
    names = ", ".join(f.label for f in unresolved) or "нет"
    return FactorProposal(
        factor=DifficultyFactor.AMBIGUITY,
        level=level,
        rationale=f"Неизвестных/неприменимых исходных фактов: {len(unresolved)} ({names}).",
    )


def _circumstances(scenario: Scenario) -> FactorProposal:
    facts = [
        f
        for f in scenario.published_facts
        if f.topic not in _DEDICATED_TOPICS and f.state is FieldState.KNOWN
    ]
    level = _level(len(facts), one=2, two=4)
    return FactorProposal(
        factor=DifficultyFactor.CIRCUMSTANCES,
        level=level,
        rationale=f"Существенных обстоятельств сверх адреса/типа/служб: {len(facts)}.",
    )


def _services(scenario: Scenario, routing: RoutingDecision | None) -> FactorProposal:
    if routing is not None and routing.services:
        count = len(routing.services)
        source = "по решению маршрутизации"
    else:
        count = 1 if any(f.topic == "services" for f in scenario.published_facts) else 0
        source = "по факту сценария (решение маршрутизации не передано)"
    level = _level(count, one=2, two=3)
    return FactorProposal(
        factor=DifficultyFactor.SERVICES,
        level=level,
        rationale=f"Вовлечённых служб: {count} ({source}).",
    )


def _additions(scenario: Scenario) -> FactorProposal:
    omissible = [f for f in scenario.published_facts if f.omissible]
    level = _level(len(omissible), one=1, two=3)
    return FactorProposal(
        factor=DifficultyFactor.ADDITIONS,
        level=level,
        rationale=f"Фактов, которые может понадобиться дополнить вручную: {len(omissible)}.",
    )


def _speech(scenario: Scenario) -> FactorProposal:
    supervisor = next((b for b in scenario.interlocutors if b.role_id == "supervisor"), None)
    if supervisor is None:
        return FactorProposal(
            factor=DifficultyFactor.SPEECH,
            level=0,
            rationale="Нет собеседника-руководителя — речевых осложнений не предусмотрено.",
        )
    if supervisor.drop_after_turns is not None:
        level, why = 2, f"обрыв связи после {supervisor.drop_after_turns} реплик"
    elif ScenarioCallState.BUSY in supervisor.dial_outcomes or (
        ScenarioCallState.NO_ANSWER in supervisor.dial_outcomes
    ):
        level, why = 1, "сценарные занято/не отвечает перед соединением"
    else:
        level, why = 0, "соединение без осложнений"
    return FactorProposal(
        factor=DifficultyFactor.SPEECH, level=level, rationale=why.capitalize() + "."
    )


def propose_vector(
    scenario: Scenario, *, routing: RoutingDecision | None = None
) -> DifficultyProposal:
    """Предложение вектора по опубликованному сценарию; преподаватель утверждает или меняет."""
    factors = (
        _ambiguity(scenario),
        _circumstances(scenario),
        _services(scenario, routing),
        _additions(scenario),
        _speech(scenario),
    )
    return DifficultyProposal(scenario=scenario.scenario, factors=factors)
