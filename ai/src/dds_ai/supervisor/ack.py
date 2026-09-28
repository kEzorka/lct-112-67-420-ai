"""Подтверждение руководителя по C-05: учёт засчитанных подтверждений по журналу событий.

Засчитывается только `ack.played_by_client` для ранее сформированного (`ack.generated`)
подтверждения того же вызова, пока вызов активен. Один `ack_id` — не более одного засчитанного
подтверждения; повторная доставка не удваивает. Слово «принято» в реплике, конец звонка,
приём входного аудио сервером подтверждения не создают. `text_delivered` — отдельный учёт
текстового режима, не голосовое подтверждение.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from ..contracts.events import (
    AckDeliveryUnconfirmed,
    AckGenerated,
    AckPlayedByClient,
    AttemptEvent,
    CallState,
    CallStateChanged,
    TextDelivered,
)

# В активном вызове ученик слышит ответ; mute выключает только его микрофон.
AUDIBLE_CALL_STATES = frozenset({CallState.ACTIVE, CallState.MUTED})
# Причина «доставка не подтверждена», которая означает действие ученика, а не сбой.
HANGUP_REASON = "call_ended_by_client"


class AckOutcome(StrEnum):
    COUNTED = "counted"
    UNCONFIRMED = (
        "unconfirmed"  # медиатракт/TTS не подтвердил доставку — техническая неопределённость
    )
    PENDING = "pending"  # сформировано, исхода доставки нет — техническая неопределённость
    PLAYED_OUTSIDE_CALL = "played_outside_active_call"
    HANGUP_BEFORE_PLAYBACK = "hangup_before_playback"


@dataclass
class AckRecord:
    ack_id: UUID
    call_id: UUID
    utterance_id: UUID
    generated_event_id: UUID
    outcome: AckOutcome = AckOutcome.PENDING
    played_event_id: UUID | None = None
    unconfirmed_reasons: list[str] = field(default_factory=list)
    repeated_deliveries: int = 0

    @property
    def technical(self) -> bool:
        return self.outcome in (
            AckOutcome.UNCONFIRMED,
            AckOutcome.PENDING,
            AckOutcome.PLAYED_OUTSIDE_CALL,
        )


@dataclass(frozen=True)
class AckSummary:
    records: tuple[AckRecord, ...]
    text_delivered_ack_ids: tuple[UUID, ...]
    orphan_played_event_ids: tuple[UUID, ...]  # воспроизведение без сформированного подтверждения

    @property
    def counted(self) -> tuple[AckRecord, ...]:
        return tuple(r for r in self.records if r.outcome is AckOutcome.COUNTED)

    def counted_for_call(self, call_id: UUID) -> int:
        return sum(1 for r in self.counted if r.call_id == call_id)

    @property
    def technical(self) -> tuple[AckRecord, ...]:
        return tuple(r for r in self.records if r.technical)


def summarize_acks(events: Iterable[AttemptEvent]) -> AckSummary:
    call_state: dict[UUID, CallState] = {}
    records: dict[UUID, AckRecord] = {}
    orphans: list[UUID] = []
    text_acks: list[UUID] = []

    for ev in sorted(events, key=lambda e: e.seq):
        if isinstance(ev, CallStateChanged):
            call_state[ev.call_id] = ev.state
        elif isinstance(ev, AckGenerated):
            if ev.ack_id not in records:  # повторное «сформировано» — тот же ack_id
                records[ev.ack_id] = AckRecord(
                    ack_id=ev.ack_id,
                    call_id=ev.call_id,
                    utterance_id=ev.utterance_id,
                    generated_event_id=ev.event_id,
                )
        elif isinstance(ev, AckPlayedByClient):
            rec = records.get(ev.ack_id)
            if rec is None or (rec.call_id, rec.utterance_id) != (ev.call_id, ev.utterance_id):
                orphans.append(ev.event_id)
            elif rec.outcome is AckOutcome.COUNTED:
                rec.repeated_deliveries += 1
            elif call_state.get(ev.call_id) in AUDIBLE_CALL_STATES:
                rec.outcome = AckOutcome.COUNTED
                rec.played_event_id = ev.event_id
            else:
                rec.outcome = AckOutcome.PLAYED_OUTSIDE_CALL
        elif isinstance(ev, AckDeliveryUnconfirmed):
            rec = records.get(ev.ack_id)
            if rec is None or rec.outcome is AckOutcome.COUNTED:
                continue
            rec.unconfirmed_reasons.append(ev.reason)
            if rec.outcome is AckOutcome.PENDING or rec.outcome is AckOutcome.UNCONFIRMED:
                rec.outcome = (
                    AckOutcome.HANGUP_BEFORE_PLAYBACK
                    if all(r == HANGUP_REASON for r in rec.unconfirmed_reasons)
                    else AckOutcome.UNCONFIRMED
                )
            elif rec.outcome is AckOutcome.HANGUP_BEFORE_PLAYBACK and ev.reason != HANGUP_REASON:
                rec.outcome = AckOutcome.UNCONFIRMED
        elif isinstance(ev, TextDelivered) and ev.ack_id is not None:
            if ev.ack_id not in text_acks:
                text_acks.append(ev.ack_id)

    return AckSummary(
        records=tuple(records.values()),
        text_delivered_ack_ids=tuple(text_acks),
        orphan_played_event_ids=tuple(orphans),
    )
