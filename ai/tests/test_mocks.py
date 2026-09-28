"""Моки соседних модулей: карточка, маршрутизация, медиатракт."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from dds_ai.contracts.card import (
    CardField,
    CardGeneration,
    CardRevisionRecord,
    FieldOrigin,
    FieldState,
    IncidentCard,
)
from dds_ai.contracts.common import VersionRef
from dds_ai.contracts.criteria import RuleStatus
from dds_ai.contracts.events import AckDeliveryUnconfirmed, AckPlayedByClient, AckSentToMedia
from dds_ai.contracts.routing import RoutingRequest
from dds_ai.mocks.card_store import ImmutableError, InMemoryCardStore
from dds_ai.mocks.media import MockMediaTransport, Playback
from dds_ai.mocks.routing import Rule, TableRoutingEngine

from .conftest import T0

OP = FieldOrigin.OPERATOR_112


def card(**fields):
    return IncidentCard(
        card_id=uuid4(),
        card_schema=VersionRef(name="card_schema", version="1"),
        created_at=T0,
        fields=fields
        or {"address": CardField(state=FieldState.KNOWN, raw="ул. Тестовая, 1", origin=OP)},
        generation=CardGeneration(scenario=VersionRef(name="synthetic-001", version="1"), seed=42),
    )


def test_unknown_none_not_applicable_are_distinct_from_empty():
    for state in (FieldState.UNKNOWN, FieldState.NONE, FieldState.NOT_APPLICABLE):
        assert CardField(state=state, origin=OP).raw is None
        with pytest.raises(ValidationError):
            CardField(state=state, raw="", origin=OP)
    with pytest.raises(ValidationError):
        CardField(state=FieldState.KNOWN, origin=OP)


def test_source_card_only_from_operator_112():
    with pytest.raises(ValidationError):
        card(address=CardField(state=FieldState.KNOWN, raw="x", origin=FieldOrigin.DISPATCHER))


def test_card_store_is_append_only():
    store = InMemoryCardStore()
    c = card()
    store.put_source(c)
    with pytest.raises(ImmutableError):
        store.put_source(c)
    edit = {"victims": CardField(state=FieldState.KNOWN, raw=2, origin=FieldOrigin.DISPATCHER)}
    store.add_revision(
        CardRevisionRecord(
            card_id=c.card_id, revision=1, author_id=uuid4(), created_at=T0, changes=edit
        )
    )
    with pytest.raises(ImmutableError):
        store.add_revision(
            CardRevisionRecord(
                card_id=c.card_id, revision=1, author_id=uuid4(), created_at=T0, changes=edit
            )
        )
    assert store.get_source(c.card_id) == c
    assert len(store.revisions(c.card_id)) == 1


def test_routing_is_deterministic_and_flags_non_active_rules():
    engine = TableRoutingEngine(
        {
            "fire": Rule(
                "r1", RuleStatus.ACTIVE, "1.1", "fire_service", ("fire_service", "ambulance")
            ),
            "odd": Rule("r2", RuleStatus.AMBIGUOUS, None, None, ()),
        }
    )
    fire = engine.route(RoutingRequest(incident_type="fire"))
    assert fire == engine.route(RoutingRequest(incident_type="fire"))
    assert fire.auto_gradable and fire.main_service == "fire_service"
    assert not engine.route(RoutingRequest(incident_type="odd")).auto_gradable
    missing = engine.route(RoutingRequest(incident_type="unknown"))
    assert (missing.rule_id, missing.rule_status, missing.auto_gradable) == (None, None, False)


@pytest.mark.parametrize(
    ("playback", "audio", "last"),
    [
        (Playback.PLAYED, b"pcm", AckPlayedByClient),
        (Playback.CALL_DROPPED, b"pcm", AckDeliveryUnconfirmed),
        (Playback.CLIENT_SILENT, b"pcm", AckDeliveryUnconfirmed),
        (Playback.PLAYED, b"", AckDeliveryUnconfirmed),
    ],
)
def test_media_emits_c05_delivery_events(playback, audio, last):
    ids = {"attempt_id": uuid4(), "call_id": uuid4(), "ack_id": uuid4(), "utterance_id": uuid4()}
    events = MockMediaTransport(playback, clock=lambda: T0).play(**ids, audio=audio)
    assert isinstance(events[0], AckSentToMedia)
    assert isinstance(events[-1], last)
    assert all(e.ack_id == ids["ack_id"] for e in events)
    assert [e.seq for e in events] == sorted(e.seq for e in events)
