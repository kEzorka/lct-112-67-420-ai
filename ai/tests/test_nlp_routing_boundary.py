"""Граница NLP ↔ маршрутизация (6.11, D-016).

`NlpExtraction` — только подсказка. Маршрут решает исключительно `mocks/routing.py`
по `RoutingRequest`; confidence, теги и прочие хинты NLP структурно не могут попасть
в `RoutingDecision`, и их изменение не меняет решение движка при одном и том же
переданном типе происшествия.
"""

from __future__ import annotations

from dds_ai.contracts.common import ModelRef
from dds_ai.contracts.criteria import RuleStatus
from dds_ai.contracts.nlp import IncidentTypeHint, NlpExtraction
from dds_ai.contracts.routing import RoutingRequest
from dds_ai.mocks.routing import Rule, TableRoutingEngine

MODEL_REF = ModelRef(component="nlp", model_name="test-model", model_version="1")


def _engine() -> TableRoutingEngine:
    rule = Rule("r1", RuleStatus.ACTIVE, "1.1", "fire_service", ("fire_service", "ambulance"))
    return TableRoutingEngine({"fire": rule})


def _route_from_top_hint(extraction: NlpExtraction):
    incident_type = extraction.top_k[0].incident_type
    return _engine().route(RoutingRequest(incident_type=incident_type))


def test_changing_nlp_confidence_and_tags_does_not_change_routing_decision():
    low_confidence = NlpExtraction(
        top_k=(IncidentTypeHint(incident_type="fire", confidence=0.3),),
        tags=("evacuation",),
        model_ref=MODEL_REF,
    )
    high_confidence = NlpExtraction(
        top_k=(IncidentTypeHint(incident_type="fire", confidence=0.95),),
        tags=(),
        model_ref=MODEL_REF,
    )
    assert _route_from_top_hint(low_confidence) == _route_from_top_hint(high_confidence)


def test_nlp_extraction_carries_no_routing_or_service_fields():
    extraction = NlpExtraction(top_k=(IncidentTypeHint(incident_type="fire"),))
    dumped = set(extraction.model_dump())
    assert dumped.isdisjoint({"ekp_code", "main_service", "services", "rule_id", "rule_status"})
