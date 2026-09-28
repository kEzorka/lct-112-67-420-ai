"""Генерация входной карточки ИИ-оператора 112 (6.5, D-003, D-042, C-07).

Базовая реализация — детерминированная (шаблон + правила): карточка строится только из
`Scenario.published_facts`. Разрешённые вариации — заранее утверждённые автором сценария
равнозначные формулировки (`ScenarioFact.variant_values`), допустимые пропуски
(`ScenarioFact.omissible`) и порядок полей; генератор не придумывает новые факты. Состояния
`unknown/none/not_applicable` сохраняются как есть, не сводятся к пустому значению
(инвариант 3). Карточка, параметры вариации и seed — в `CardGeneration` (D-003): то же
(версия сценария, seed) даёт ту же карточку (C-07).

Опциональный путь — переформулировка описания через порт `LLMProvider`/`FaultInjector`
(6.5.5): выход проверяется на отсутствие чисел вне опубликованных фактов (инвариант 5,
«нет выдуманных чисел»); отказ модели или невалидный выход — откат к детерминированному
варианту (C-04), режим фиксируется в `variation_params`.
"""

from __future__ import annotations

import random
from datetime import datetime
from uuid import UUID, uuid4

from .. import facts_guard
from ..contracts.card import CardField, CardGeneration, FieldOrigin, FieldState, IncidentCard
from ..contracts.common import VersionRef
from ..contracts.events import ComponentName
from ..contracts.scenario import Scenario, ScenarioFact
from ..faults import ComponentFailure, FaultInjector, InvalidOutput
from ..ports import LLMProvider

CARD_SCHEMA = VersionRef(name="card_schema", version="draft-1")
DESCRIPTION_TOPIC = "description"


def _rng(scenario: Scenario, seed: int) -> random.Random:
    return random.Random(f"{scenario.scenario.name}:{scenario.scenario.version}:{seed}")


def _allowed_values(fact: ScenarioFact) -> tuple[str, ...]:
    if fact.state is not FieldState.KNOWN:
        return ()
    assert fact.value is not None
    return (fact.value, *fact.variant_values)


def generate_card(
    scenario: Scenario,
    *,
    seed: int,
    created_at: datetime,
    card_id: UUID | None = None,
) -> IncidentCard:
    """Детерминированная генерация карточки: чистая функция (сценарий, seed) (D-003, C-07)."""
    rng = _rng(scenario, seed)
    by_topic = {f.topic: f for f in scenario.published_facts}
    order = list(by_topic)
    rng.shuffle(order)

    omitted: list[str] = []
    chosen_variant: dict[str, str] = {}
    fields: dict[str, CardField] = {}
    for topic in order:
        fact = by_topic[topic]
        state = fact.state
        values = _allowed_values(fact)
        value = rng.choice(values) if values else None
        if value is not None and value != fact.value:
            chosen_variant[topic] = value
        if fact.omissible and state is FieldState.KNOWN and rng.random() < 0.5:
            state, value = FieldState.UNKNOWN, None
            omitted.append(topic)
        fields[topic] = CardField(state=state, raw=value, origin=FieldOrigin.OPERATOR_112)

    variation_params: dict[str, object] = {"field_order": order}
    if omitted:
        variation_params["omitted"] = omitted
    if chosen_variant:
        variation_params["chosen_variant"] = chosen_variant

    return IncidentCard(
        card_id=card_id or uuid4(),
        card_schema=CARD_SCHEMA,
        created_at=created_at,
        fields=fields,
        generation=CardGeneration(
            scenario=scenario.scenario, variation_params=variation_params, seed=seed
        ),
    )


def _known_facts_text(scenario: Scenario) -> str:
    parts: list[str] = []
    for fact in scenario.published_facts:
        parts.append(fact.label)
        parts.extend(_allowed_values(fact))
    return "; ".join(parts)


def _validate_description(text: str, scenario: Scenario) -> str:
    """Отклонить переформулировку, вносящую факты вне опубликованных фактов сценария
    (инвариант 5): числа, адресные элементы, службы, имена — общая проверка facts_guard (E-5)."""
    text = text.strip()
    if not text:
        raise InvalidOutput("empty description")
    violation = facts_guard.find_violation(text, allowed=_known_facts_text(scenario))
    if violation is not None:
        raise InvalidOutput(f"description invents a fact outside scenario: {violation.kind}")
    return text


def _describe_prompt(scenario: Scenario, current: str) -> str:
    facts = "; ".join(
        f"{f.label}: {f.value}" for f in scenario.published_facts if f.state is FieldState.KNOWN
    )
    return (
        "Перефразируй описание происшествия для карточки диспетчера ДДС, не добавляя фактов "
        f"сверх перечисленных. Факты: {facts}. Текущее описание: {current}"
    )


def generate_card_with_llm(
    scenario: Scenario,
    *,
    seed: int,
    created_at: datetime,
    llm: LLMProvider,
    injector: FaultInjector,
    card_id: UUID | None = None,
) -> IncidentCard:
    """Как `generate_card`, но описание может переформулировать LLM (6.5.5).

    Выход валидируется; отказ модели, тайм-аут, переполнение или невалидный выход
    (`ComponentFailure`, включая враждебный ответ с выдуманным фактом) — откат к
    детерминированной карточке без переформулировки (C-04).
    """
    base = generate_card(scenario, seed=seed, created_at=created_at, card_id=card_id)
    description = base.fields.get(DESCRIPTION_TOPIC)
    if description is None or description.state is not FieldState.KNOWN:
        return base
    prompt = _describe_prompt(scenario, str(description.raw))
    try:
        text = injector.call(
            ComponentName.LLM,
            lambda: _validate_description(llm.complete(prompt, max_tokens=200), scenario),
        )
    except ComponentFailure:
        return base

    fields = dict(base.fields)
    fields[DESCRIPTION_TOPIC] = CardField(
        state=FieldState.KNOWN, raw=text, origin=FieldOrigin.OPERATOR_112
    )
    variation_params = dict(base.generation.variation_params)
    variation_params["description_mode"] = "llm"
    return IncidentCard(
        card_id=base.card_id,
        card_schema=base.card_schema,
        created_at=base.created_at,
        fields=fields,
        generation=CardGeneration(
            scenario=base.generation.scenario,
            variation_params=variation_params,
            seed=seed,
            model_ref_id=f"{llm.model_ref.component}:{llm.model_ref.model_name}:"
            f"{llm.model_ref.model_version}",
        ),
    )
