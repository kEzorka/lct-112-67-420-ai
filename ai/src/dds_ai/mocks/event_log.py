"""Мок журнала событий попытки (бэкенд): append-only, сервер назначает seq и server_ts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from uuid import UUID, uuid4

from ..contracts.events import AttemptEvent, EventSource

E = TypeVar("E")


class ManualClock:
    """Управляемые часы для воспроизводимых временных тестов (C-01)."""

    def __init__(self, start: datetime | None = None):
        self._now = start or datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> datetime:
        self._now += timedelta(seconds=seconds)
        return self._now


class InMemoryEventLog:
    def __init__(self, attempt_id: UUID, clock: Callable[[], datetime] | None = None):
        self.attempt_id = attempt_id
        self._clock = clock or (lambda: datetime.now(UTC))
        self._events: list[AttemptEvent] = []

    @property
    def events(self) -> tuple[AttemptEvent, ...]:
        return tuple(self._events)

    def append(self, event_cls: type[E], *, source: EventSource, **fields: Any) -> E:
        event = event_cls(
            event_id=uuid4(),
            attempt_id=self.attempt_id,
            seq=len(self._events) + 1,
            server_ts=self._clock(),
            source=source,
            **fields,
        )
        self._events.append(event)
        return event

    def ingest(self, event: AttemptEvent) -> AttemptEvent:
        """Принять событие другого компонента (медиатракт): seq и время назначает журнал."""
        if event.attempt_id != self.attempt_id:
            raise ValueError("event belongs to another attempt")
        stored = event.model_copy(update={"seq": len(self._events) + 1, "server_ts": self._clock()})
        self._events.append(stored)
        return stored

    def of_type(self, *types: type) -> list[AttemptEvent]:
        return [e for e in self._events if isinstance(e, types)]
