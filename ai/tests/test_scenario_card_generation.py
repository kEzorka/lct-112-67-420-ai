"""Генерация карточки ИИ-оператора 112 (6.5): только опубликованные факты, вариация,
воспроизводимость по seed, откат LLM при выдуманном факте (инвариант 5, «галлюцинации»)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dds_ai.contracts.card import FieldState
from dds_ai.contracts.common import ModelRef, VersionRef
from dds_ai.contracts.dialogue import Interlocutor, SupervisorAction
from dds_ai.contracts.events import ComponentName, DispatcherDecision, FailureKind
from dds_ai.contracts.scenario import (
    InterlocutorBrief,
    Provenance,
    Scenario,
    ScenarioFact,
    ScenarioReference,
)
from dds_ai.faults import FaultInjector
from dds_ai.scenarios.card_generation import generate_card, generate_card_with_llm

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _scenario(version: str = "1", *, omissible_floors: bool = True) -> Scenario:
    return Scenario(
        scenario=VersionRef(name="syn-test-card", version=version),
        title="СИНТЕТИКА: тест генерации карточки",
        provenance=Provenance.SYNTHETIC,
        incident_type="fire_residential",
        published_facts=(
            ScenarioFact(
                fact_id="card.address",
                topic="address",
                label="Адрес",
                state=FieldState.KNOWN,
                value="г. Учебный, ул. Тестовая, д. 12",
                variant_values=("ул. Тестовая, дом 12, г. Учебный",),
            ),
            ScenarioFact(
                fact_id="card.incident_type",
                topic="incident_type",
                label="Тип происшествия",
                state=FieldState.KNOWN,
                value="пожар в жилом доме",
            ),
            ScenarioFact(
                fact_id="card.description",
                topic="description",
                label="Описание",
                state=FieldState.KNOWN,
                value="Заявитель видит дым из окна квартиры соседей",
            ),
            ScenarioFact(
                fact_id="card.victims",
                topic="victims",
                label="Пострадавшие",
                state=FieldState.UNKNOWN,
            ),
            ScenarioFact(
                fact_id="card.floors",
                topic="floors",
                label="Этажность",
                state=FieldState.KNOWN,
                value="9 этажей",
                omissible=omissible_floors,
            ),
        ),
        interlocutors=(
            InterlocutorBrief(
                interlocutor=Interlocutor(
                    role_id="supervisor", allowed_actions=(SupervisorAction.LISTEN,)
                ),
            ),
        ),
        reference=ScenarioReference(
            reference=VersionRef(name="syn-test-card.ref", version="1"),
            expected_decision=DispatcherDecision.RESPOND,
            expected_fields={"address": "г. Учебный, ул. Тестовая, д. 12"},
        ),
    )


# --- детерминированный генератор ------------------------------------------------------------


def test_same_seed_and_version_give_the_same_card():
    scenario = _scenario()
    a = generate_card(scenario, seed=7, created_at=T0)
    b = generate_card(scenario, seed=7, created_at=T0)
    assert a.model_dump(exclude={"card_id"}) == b.model_dump(exclude={"card_id"})


def test_different_seed_can_give_a_different_card():
    scenario = _scenario()
    # хотя бы одна вариация (пропуск этажности) должна встретиться среди 20 seed
    omitted_seen = any(
        generate_card(scenario, seed=s, created_at=T0).fields["floors"].state is FieldState.UNKNOWN
        for s in range(20)
    )
    kept_seen = any(
        generate_card(scenario, seed=s, created_at=T0).fields["floors"].state is FieldState.KNOWN
        for s in range(20)
    )
    assert omitted_seen and kept_seen


def test_card_contains_only_scenario_facts_across_seeds():
    scenario = _scenario()
    allowed = {
        "address": {"г. Учебный, ул. Тестовая, д. 12", "ул. Тестовая, дом 12, г. Учебный"},
        "incident_type": {"пожар в жилом доме"},
        "description": {"Заявитель видит дым из окна квартиры соседей"},
        "floors": {"9 этажей"},
    }
    for seed in range(20):
        card = generate_card(scenario, seed=seed, created_at=T0)
        for topic, field in card.fields.items():
            if field.state is FieldState.KNOWN:
                assert field.raw in allowed[topic]
            else:
                assert field.raw is None


def test_unknown_state_is_never_reduced_to_empty_value():
    scenario = _scenario()
    card = generate_card(scenario, seed=1, created_at=T0)
    victims = card.fields["victims"]
    assert victims.state is FieldState.UNKNOWN
    assert victims.raw is None


def test_variant_values_are_pre_authored_not_invented():
    """Все варианты значения — то, что заранее утвердил автор сценария (variant_values),
    генератор не подставляет ничего вне этого множества."""
    scenario = _scenario()
    seen = {generate_card(scenario, seed=s, created_at=T0).fields["address"].raw for s in range(30)}
    assert seen <= {"г. Учебный, ул. Тестовая, д. 12", "ул. Тестовая, дом 12, г. Учебный"}
    assert len(seen) == 2  # оба варианта встречаются на достаточном числе seed


def test_generation_is_recorded_for_reproduction():
    scenario = _scenario()
    card = generate_card(scenario, seed=42, created_at=T0)
    assert card.generation.scenario == scenario.scenario
    assert card.generation.seed == 42
    assert isinstance(card.generation.variation_params, dict)


def test_changing_the_scenario_object_does_not_change_an_already_generated_card():
    """Изменение опубликованного сценария не меняет начатую попытку (6.5, D-003):
    генерация — чистая функция снимка сценария в момент вызова."""
    scenario_v1 = _scenario(version="1")
    card = generate_card(scenario_v1, seed=3, created_at=T0)
    scenario_v2 = _scenario(version="2", omissible_floors=False)
    # тот же вызов с новой версией сценария не переписывает уже сгенерированную карточку
    generate_card(scenario_v2, seed=3, created_at=T0)
    assert card.generation.scenario.version == "1"
    assert card.fields["address"].raw in {
        "г. Учебный, ул. Тестовая, д. 12",
        "ул. Тестовая, дом 12, г. Учебный",
    }


# --- LLM-вариант описания + защита от выдуманных фактов ---------------------------------------


class _FakeLLM:
    def __init__(self, text: str):
        self.model_ref = ModelRef(component="llm", model_name="fake-llm", model_version="0")
        self.text = text
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        self.calls += 1
        return self.text


def test_llm_paraphrase_within_known_facts_is_accepted():
    scenario = _scenario()
    llm = _FakeLLM("Сосед сообщает: из окна квартиры соседей идёт дым, дом 9-этажный.")
    injector = FaultInjector()
    card = generate_card_with_llm(scenario, seed=1, created_at=T0, llm=llm, injector=injector)
    assert card.fields["description"].raw == llm.text
    assert card.generation.variation_params["description_mode"] == "llm"
    assert card.generation.model_ref_id is not None
    assert not injector.log


def test_hostile_llm_inventing_a_new_fact_is_rejected():
    """Враждебный фейк LLM добавляет новый факт (число пострадавших) — карточка его не несёт,
    генератор откатывается к детерминированному варианту (инвариант 5, C-04)."""
    scenario = _scenario()
    hostile = _FakeLLM("Пострадало 3 человека, всех госпитализировали.")
    injector = FaultInjector()
    fallback = generate_card(scenario, seed=9, created_at=T0)
    card = generate_card_with_llm(scenario, seed=9, created_at=T0, llm=hostile, injector=injector)
    assert card.fields["description"].raw != hostile.text
    assert card.fields["description"].raw == fallback.fields["description"].raw
    assert card.generation.variation_params.get("description_mode") != "llm"
    assert injector.log and injector.log[-1].kind is FailureKind.INVALID_OUTPUT
    assert injector.log[-1].component is ComponentName.LLM
    # враждебный факт нигде не просочился в карточку
    dumped = card.model_dump_json()
    assert "3 человека" not in dumped
    assert "госпитализ" not in dumped


def test_llm_outage_falls_back_to_deterministic_card():
    scenario = _scenario()
    injector = FaultInjector()
    injector.inject(ComponentName.LLM, FailureKind.TIMEOUT)
    llm = _FakeLLM("не важно")
    fallback = generate_card(scenario, seed=5, created_at=T0)
    card = generate_card_with_llm(scenario, seed=5, created_at=T0, llm=llm, injector=injector)
    assert card.fields["description"].raw == fallback.fields["description"].raw
    assert llm.calls == 0  # инъекция срабатывает раньше самого вызова модели
    assert injector.log[-1].kind is FailureKind.TIMEOUT


@pytest.mark.parametrize("hostile_text", ["", "   "])
def test_empty_llm_output_falls_back(hostile_text: str):
    scenario = _scenario()
    llm = _FakeLLM(hostile_text)
    injector = FaultInjector()
    fallback = generate_card(scenario, seed=2, created_at=T0)
    card = generate_card_with_llm(scenario, seed=2, created_at=T0, llm=llm, injector=injector)
    assert card.fields["description"].raw == fallback.fields["description"].raw
