"""Генератор структурированного эталона (6.5.3): привязка к рубрике и источникам,
воспроизводимость из сохранённых версий (сценарий + рубрика + фрагменты → тот же эталон)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from dds_ai.contracts.card import FieldState
from dds_ai.contracts.common import VersionRef
from dds_ai.contracts.dialogue import Interlocutor, SupervisorAction
from dds_ai.contracts.events import DispatcherDecision
from dds_ai.contracts.knowledge import ApprovalStatus, FragmentSource, KnowledgeFragment
from dds_ai.contracts.rubric import Rubric
from dds_ai.contracts.scenario import (
    InterlocutorBrief,
    Provenance,
    Scenario,
    ScenarioFact,
    ScenarioReference,
)
from dds_ai.scenarios.reference_generation import generate_reference

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
CONFIG = Path(__file__).resolve().parents[1] / "config"


def _rubric() -> Rubric:
    return Rubric.model_validate(json.loads((CONFIG / "rubric.w01.json").read_text("utf-8")))


def _scenario() -> Scenario:
    return Scenario(
        scenario=VersionRef(name="gen-reference-test", version="1"),
        title="СИНТЕТИКА: тест генерации эталона",
        provenance=Provenance.SYNTHETIC,
        incident_type="fire_residential",
        published_facts=(
            ScenarioFact(
                fact_id="card.address",
                topic="address",
                label="Адрес",
                state=FieldState.KNOWN,
                value="г. Учебный, ул. Тестовая, д. 1",
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
            reference=VersionRef(name="gen-reference-test.ref", version="1"),
            expected_decision=DispatcherDecision.RESPOND,
            expected_fields={"address": "г. Учебный, ул. Тестовая, д. 1"},
        ),
    )


def _fragment(fragment_id: str) -> KnowledgeFragment:
    return KnowledgeFragment(
        fragment_id=fragment_id,
        corpus=VersionRef(name="knowledge_corpus", version="1"),
        source=FragmentSource(
            doc_id="doc-1",
            title="Тестовый документ",
            source="synthetic",
            approval_status=ApprovalStatus.APPROVED,
        ),
        chunk_index=0,
        text="Тестовый синтетический фрагмент.",
        created_at=T0,
    )


def test_reference_links_every_rubric_criterion_to_a_group():
    rubric = _rubric()
    reference = generate_reference(_scenario(), rubric=rubric, new_version="1")
    linked_ids = {link.criterion_id for link in reference.criteria_links}
    all_ids = {c.criterion_id for g in rubric.groups for c in g.criteria}
    assert linked_ids == all_ids
    for link in reference.criteria_links:
        group = rubric.group_of(link.criterion_id)
        assert link.group_id == group.group_id
        assert link.rationale


def test_reference_carries_used_fragment_ids():
    rubric = _rubric()
    fragments = (_fragment("doc-1::1::0"), _fragment("doc-1::1::1"))
    reference = generate_reference(
        _scenario(), rubric=rubric, new_version="1", fragments_used=fragments
    )
    for link in reference.criteria_links:
        assert set(link.source_fragment_ids) == {"doc-1::1::0", "doc-1::1::1"}


def test_reference_preserves_expected_decision_and_fields():
    scenario = _scenario()
    reference = generate_reference(scenario, rubric=_rubric(), new_version="2")
    assert reference.expected_decision == scenario.reference.expected_decision
    assert reference.expected_fields == scenario.reference.expected_fields
    assert reference.reference.version == "2"
    assert reference.reference.name == scenario.reference.reference.name


def test_reference_is_reproducible_from_saved_versions():
    """Тот же (сценарий, рубрика, версия, фрагменты) → тот же эталон: воспроизводимость (6.5)."""
    scenario = _scenario()
    rubric = _rubric()
    fragments = (_fragment("doc-1::1::0"),)
    a = generate_reference(scenario, rubric=rubric, new_version="3", fragments_used=fragments)
    b = generate_reference(scenario, rubric=rubric, new_version="3", fragments_used=fragments)
    assert a == b


def test_correcting_the_reference_creates_a_new_version_not_overwriting_the_old():
    scenario = _scenario()
    rubric = _rubric()
    v1 = generate_reference(scenario, rubric=rubric, new_version="1")
    v2 = generate_reference(scenario, rubric=rubric, new_version="2")
    assert v1.reference.version == "1"
    assert v2.reference.version == "2"
    assert v1 != v2  # разные версии — разные объекты, старый не переписан (C-09)


def test_routing_services_rationale_points_to_the_rule_engine_not_the_reference():
    reference = generate_reference(_scenario(), rubric=_rubric(), new_version="1")
    link = next(
        item for item in reference.criteria_links if item.criterion_id == "routing.services"
    )
    assert "движк" in link.rationale.lower() or "правил" in link.rationale.lower()
