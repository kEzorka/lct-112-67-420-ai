"""Журнал событий попытки: дискриминированный union, строгая схема."""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from dds_ai.contracts.events import (
    AckPlayedByClient,
    CardOpened,
    SelectedAction,
    Timeout,
    attempt_event_adapter,
    parse_event,
)

from .conftest import T0


def base(seq=1, **kw):
    return {
        "event_id": str(uuid4()),
        "attempt_id": str(uuid4()),
        "seq": seq,
        "server_ts": T0.isoformat(),
        **kw,
    }


def test_parses_by_type():
    e = parse_event(base(type="card_opened", source="client", card_id=str(uuid4())))
    assert isinstance(e, CardOpened)
    e = parse_event(base(type="selected_action", source="client", decision="redirect"))
    assert isinstance(e, SelectedAction)


def test_played_audio_ack_is_the_same_as_played_by_client():
    ids = {"ack_id": str(uuid4()), "call_id": str(uuid4()), "utterance_id": str(uuid4())}
    e = parse_event(base(type="ack.played_by_client", source="media", **ids))
    assert isinstance(e, AckPlayedByClient)
    with pytest.raises(ValidationError):
        parse_event(base(type="played_audio_ack", source="media", **ids))


def test_timeout_is_only_an_event():
    e = parse_event(base(type="timeout", source="backend", limit="processing"))
    assert isinstance(e, Timeout)
    assert set(Timeout.model_fields) == {
        "event_id",
        "attempt_id",
        "seq",
        "server_ts",
        "client_ts",
        "source",
        "type",
        "limit",
    }


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError):
        parse_event(
            base(type="notification_shown", source="client")
            | {"server_ts": datetime(2026, 1, 1).isoformat()}
        )


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        parse_event(base(type="notification_shown", source="client", score=100))


def test_json_roundtrip():
    e = parse_event(base(type="model_failure", source="ai_worker", component="stt", kind="timeout"))
    again = parse_event(attempt_event_adapter.dump_json(e))
    assert again == e
