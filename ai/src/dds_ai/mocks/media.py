"""Мок медиатракта VoIP-эмулятора: генерирует события доставки подтверждения (C-05)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from ..contracts.events import (
    AckDeliveryUnconfirmed,
    AckPlayedByClient,
    AckSentToMedia,
    AttemptEvent,
    EventSource,
)


class Playback(StrEnum):
    PLAYED = "played"
    CALL_DROPPED = "call_dropped"
    CLIENT_SILENT = "client_silent"


class MockMediaTransport:
    """seq назначает бэкенд; мок нумерует события сам, начиная с start_seq."""

    def __init__(
        self,
        playback: Playback = Playback.PLAYED,
        start_seq: int = 1000,
        clock: Callable[[], datetime] | None = None,
    ):
        self.playback = playback
        self._seq = start_seq
        self._clock = clock or (lambda: datetime.now(UTC))

    def _next(self) -> dict:
        self._seq += 1
        return {
            "event_id": uuid4(),
            "seq": self._seq,
            "server_ts": self._clock(),
            "source": EventSource.MEDIA,
        }

    def play(
        self, *, attempt_id: UUID, call_id: UUID, ack_id: UUID, utterance_id: UUID, audio: bytes
    ) -> list[AttemptEvent]:
        ids = {
            "attempt_id": attempt_id,
            "call_id": call_id,
            "ack_id": ack_id,
            "utterance_id": utterance_id,
        }
        events: list[AttemptEvent] = [AckSentToMedia(**self._next(), **ids)]
        if self.playback is Playback.PLAYED and audio:
            events.append(AckPlayedByClient(**self._next(), **ids))
        else:
            reason = "empty audio" if not audio else self.playback.value
            events.append(AckDeliveryUnconfirmed(**self._next(), **ids, reason=reason))
        return events
