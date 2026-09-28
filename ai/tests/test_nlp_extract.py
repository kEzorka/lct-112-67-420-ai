"""NLP-извлечение признаков (6.11): top-k, слоты, пропуски, противоречия, эквивалентность."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dds_ai.contracts.nlp import IncidentTypeHint, NlpExtraction, SlotState
from dds_ai.nlp import extract_features
from dds_ai.nlp.slots import extract_address, extract_victims


def _slot(extraction: NlpExtraction, name: str):
    return next(s for s in extraction.slots if s.name == name)


def test_missing_address_is_flagged():
    result = extract_features("Пожар в здании, идёт дым, есть пострадавшие.")
    assert any(m.slot == "address" for m in result.missing)
    assert _slot(result, "address").state is SlotState.UNKNOWN
    assert result.needs_clarification


def test_address_present_is_not_flagged_as_missing():
    result = extract_features("Пожар по адресу ул. Ленина, д. 10.")
    assert not any(m.slot == "address" for m in result.missing)
    assert _slot(result, "address").state is SlotState.KNOWN


def test_contradictory_victims_count_is_flagged():
    text = "Пожар, два пострадавших. Позже уточнили: пострадавших трое."
    result = extract_features(text)
    assert len(result.contradictions) == 1
    flag = result.contradictions[0]
    assert flag.slot == "victims_count"
    assert len(flag.evidence) == 2
    # Само значение слота не подставляется наугад при противоречии (инвариант 5).
    assert _slot(result, "victims_count").state is SlotState.UNKNOWN
    assert result.needs_clarification


def test_consistent_victims_count_is_not_a_contradiction():
    text = "Пожар, два пострадавших. Пострадавших госпитализировали, двое пострадавших живы."
    result = extract_features(text)
    assert result.contradictions == ()
    assert _slot(result, "victims_count").value == "2"


def test_no_victims_mentioned_explicitly_is_known_zero():
    result = extract_features("Пожар в здании по адресу ул. Ленина, д. 10, пострадавших нет.")
    assert _slot(result, "victims_count").value == "0"


@pytest.mark.parametrize(
    ("text_a", "text_b"),
    [
        (
            "Пожар по адресу ул. Ленина, д. 10. Видно дым, два человека пострадали, "
            "идёт эвакуация.",
            "Загорелось на улице Ленина, дом 10. Пострадавших двое, дым, жильцов эвакуируют.",
        ),
    ],
)
def test_paraphrasing_the_same_facts_gives_the_same_slots(text_a: str, text_b: str):
    a = extract_features(text_a)
    b = extract_features(text_b)
    values_a = sorted((s.name, s.state, s.value) for s in a.slots)
    values_b = sorted((s.name, s.state, s.value) for s in b.slots)
    assert values_a == values_b


def test_top_k_incident_types_ranked_by_keyword_signal():
    result = extract_features("Пожар, дым, огонь, горит проводка. Также слышен запах газа.")
    types = [h.incident_type for h in result.top_k]
    assert types[0] == "fire"
    assert "gas_leak" in types


def test_rule_based_extraction_never_carries_confidence():
    result = extract_features("Пожар в здании.")
    assert all(h.confidence is None for h in result.top_k)
    assert result.model_ref is None


def test_extraction_output_does_not_assign_routing_fields():
    result = extract_features("Пожар, дым, пострадавших нет.")
    dumped = result.model_dump()
    assert "ekp_code" not in dumped
    assert "main_service" not in dumped
    assert "services" not in dumped


def test_rule_based_hint_with_confidence_is_rejected():
    with pytest.raises(ValidationError):
        NlpExtraction(top_k=(IncidentTypeHint(incident_type="fire", confidence=0.9),))


def test_extract_address_returns_unknown_without_match():
    slot = extract_address("Происшествие без указания адреса.")
    assert slot.state is SlotState.UNKNOWN
    assert slot.value is None


def test_extract_victims_returns_unknown_without_mentions():
    slot, contradiction = extract_victims("Происшествие без сведений о пострадавших.")
    assert slot.state is SlotState.UNKNOWN
    assert contradiction is None
