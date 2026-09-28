"""Контракт сценария (M1) и синтетический набор."""

import pytest
from pydantic import ValidationError

from dds_ai.contracts import SCHEMA_MODELS
from dds_ai.contracts.dialogue import ScenarioCallState
from dds_ai.contracts.events import DispatcherDecision
from dds_ai.contracts.scenario import (
    SYNTHETIC_TITLE_PREFIX,
    InterlocutorBrief,
    Provenance,
    Scenario,
    ScenarioFact,
)
from dds_ai.cycle import load_synthetic


@pytest.fixture(scope="module")
def scenarios():
    return load_synthetic()


def test_synthetic_set_is_marked_and_covers_all_decisions(scenarios):
    # 8-12 сценариев для 2-3 профилей ДДС (6.5); точное число растёт с набором, не фиксировано
    assert 8 <= len(scenarios) <= 12
    for s in scenarios.values():
        assert s.provenance is Provenance.SYNTHETIC
        assert s.title.startswith(SYNTHETIC_TITLE_PREFIX)
        assert s.source_ref is None
    assert {s.reference.expected_decision for s in scenarios.values()} == set(DispatcherDecision)


def test_synthetic_set_covers_scenario_call_states(scenarios):
    briefs = [s.brief("supervisor") for s in scenarios.values()]
    states = {o for b in briefs for o in b.dial_outcomes}
    assert {ScenarioCallState.BUSY, ScenarioCallState.NO_ANSWER} <= states
    assert any(b.drop_after_turns for b in briefs)


def test_scenario_is_registered_for_schema_export():
    assert SCHEMA_MODELS["scenario"] is Scenario


def test_synthetic_cannot_pose_as_ticket(scenarios):
    data = scenarios["syn-001-fire-respond"].model_dump(mode="json")
    with pytest.raises(ValidationError):
        Scenario.model_validate({**data, "title": "Билет 17: пожар"})
    with pytest.raises(ValidationError):
        Scenario.model_validate({**data, "source_ref": "ticket-17"})
    with pytest.raises(ValidationError):
        Scenario.model_validate({**data, "provenance": "ticket"})  # билет без source_ref


def test_unknown_fact_carries_no_value():
    with pytest.raises(ValidationError):
        ScenarioFact(fact_id="f", topic="victims", label="Пострадавшие", state="unknown", value="0")
    with pytest.raises(ValidationError):
        ScenarioFact(fact_id="f", topic="victims", label="Пострадавшие", state="known")


def test_brief_facts_match_interlocutor_contract(scenarios):
    brief = scenarios["syn-001-fire-respond"].brief("supervisor")
    data = brief.model_dump(mode="json")
    data["interlocutor"]["published_fact_ids"] = ["sv.victims"]  # sv.floors не опубликован
    with pytest.raises(ValidationError):
        InterlocutorBrief.model_validate(data)


def test_report_items_cover_every_element(scenarios):
    data = scenarios["syn-001-fire-respond"].brief("supervisor").model_dump(mode="json")
    data["report_items"] = [i for i in data["report_items"] if i["element"] != "location"]
    with pytest.raises(ValidationError):
        InterlocutorBrief.model_validate(data)


def test_interlocutor_contract_has_role_id_and_operator_112_is_off(scenarios):
    s = scenarios["syn-001-fire-respond"]
    assert s.brief("supervisor").role_id == "supervisor"
    assert s.brief("operator_112").enabled is False  # D-030


def test_brief_does_not_carry_reference(scenarios):
    brief = scenarios["syn-001-fire-respond"].brief("supervisor")
    dumped = brief.model_dump_json()
    assert "expected_decision" not in dumped and "expected_fields" not in dumped
    assert "reference" not in InterlocutorBrief.model_fields
