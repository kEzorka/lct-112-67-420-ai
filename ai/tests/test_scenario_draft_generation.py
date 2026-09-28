"""Генерация черновика сценария для преподавателя (6.5.2): только из входных параметров и
утверждённых фрагментов базы знаний; число пострадавших без явного указания остаётся
`unknown` (инвариант 3); интеграция с `KnowledgeIndex` (6.10)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dds_ai.contracts.card import FieldState
from dds_ai.contracts.events import DispatcherDecision
from dds_ai.contracts.scenario_lifecycle import ScenarioStatus
from dds_ai.faults import FaultInjector
from dds_ai.knowledge.embedder import HashingEmbedder
from dds_ai.knowledge.pipeline import build_corpus, build_index
from dds_ai.scenarios.draft_generation import generate_draft
from dds_ai.scenarios.lifecycle import validate_structure

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "knowledge"


def test_generated_draft_is_a_draft_and_structurally_valid():
    record = generate_draft(
        name="gen-draft-1",
        version="1",
        incident_type="пожар в жилом доме",
        location="г. Учебный, ул. Северная, д. 5",
        difficulty="medium",
        topic="Заявитель видит дым из окна",
        dds_profile="dds-center",
        expected_decision=DispatcherDecision.RESPOND,
        created_at=T0,
    )
    assert record.status is ScenarioStatus.DRAFT
    assert record.scenario.title.startswith("СИНТЕТИКА")
    assert not validate_structure(record.scenario)


def test_victims_default_to_unknown_not_invented():
    record = generate_draft(
        name="gen-draft-2",
        version="1",
        incident_type="запах газа",
        location="г. Учебный, пр. Условный, д. 7",
        difficulty="easy",
        topic="Запах газа на площадке",
        dds_profile="dds-center",
        expected_decision=DispatcherDecision.REDIRECT,
        created_at=T0,
    )
    victims = next(f for f in record.scenario.published_facts if f.topic == "victims")
    assert victims.state is FieldState.UNKNOWN
    assert victims.value is None


def test_explicit_victims_value_is_honored():
    record = generate_draft(
        name="gen-draft-3",
        version="1",
        incident_type="пожар в жилом доме",
        location="г. Учебный, ул. Южная, д. 2",
        difficulty="hard",
        topic="Пожар с пострадавшими",
        dds_profile="dds-north",
        expected_decision=DispatcherDecision.RESPOND,
        created_at=T0,
        victims_state=FieldState.KNOWN,
        victims_value="2 человека",
    )
    victims = next(f for f in record.scenario.published_facts if f.topic == "victims")
    assert victims.state is FieldState.KNOWN
    assert victims.value == "2 человека"


def test_diagnostic_draft_marks_routing_not_applicable():
    record = generate_draft(
        name="gen-draft-4",
        version="1",
        incident_type="неклассифицированный запах",
        location="г. Учебный, ул. Дальняя, д. 9",
        difficulty="medium",
        topic="Неопределённый запах в подвале",
        dds_profile="dds-center",
        expected_decision=DispatcherDecision.REFUSE,
        created_at=T0,
        diagnostic=True,
    )
    assert record.scenario.diagnostic is True
    assert "routing.services" in record.scenario.reference.not_applicable_criteria


def test_draft_uses_knowledge_index_and_records_fragments_and_corpus_version():
    corpus = build_corpus(KNOWLEDGE_DIR, version="test-1", now=T0)
    index = build_index(corpus, HashingEmbedder(), FaultInjector())
    record = generate_draft(
        name="gen-draft-5",
        version="1",
        incident_type="пожар в жилом доме",
        location="г. Учебный, ул. Дымная, д. 4",
        difficulty="medium",
        topic="Пожар, эвакуация жильцов",
        dds_profile="dds-center",
        expected_decision=DispatcherDecision.RESPOND,
        created_at=T0,
        knowledge_index=index,
    )
    assert record.generation.corpus_version == corpus.version.ref
    assert record.generation.used_fragment_ids
    for fragment_id in record.generation.used_fragment_ids:
        assert any(f.fragment_id == fragment_id for f in corpus.fragments)


def test_draft_without_knowledge_index_has_no_fragment_refs():
    record = generate_draft(
        name="gen-draft-6",
        version="1",
        incident_type="пожар в жилом доме",
        location="г. Учебный, ул. Тихая, д. 3",
        difficulty="easy",
        topic="Дым в подъезде",
        dds_profile="dds-center",
        expected_decision=DispatcherDecision.RESPOND,
        created_at=T0,
    )
    assert record.generation.used_fragment_ids == ()
    assert record.generation.corpus_version is None
    assert record.generation.template_version is not None
    assert record.generation.model_ref is None  # базовая реализация детерминированная (6.5.5)


def test_report_items_cover_all_elements_and_match_the_decision():
    from dds_ai.contracts.scenario import ReportElement

    record = generate_draft(
        name="gen-draft-7",
        version="1",
        incident_type="запах газа",
        location="г. Учебный, ул. Газовая, д. 1",
        difficulty="medium",
        topic="Запах газа в квартире",
        dds_profile="dds-center",
        expected_decision=DispatcherDecision.REDIRECT,
        created_at=T0,
    )
    items = record.scenario.interlocutors[0].report_items
    covered = {i.element for i in items}
    assert covered == set(ReportElement)
    decision_item = next(i for i in items if i.element is ReportElement.DDS_DECISION)
    assert any("перенаправ" in p for p in decision_item.patterns)
