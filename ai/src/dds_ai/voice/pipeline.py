"""Голосовой пайплайн через мок медиатракта (M2): mute/hold, STT/TTS через FaultInjector,
доставка подтверждения руководителя по C-05.

Инварианты, которые обеспечивает этот модуль:
- в mute и на удержании звук ученика не поступает в STT (6.2);
- отказ STT/TTS идёт через FaultInjector и даёт помеченный текстовый режим, а не тишину
  и не выдуманный результат (инвариант 4, C-04);
- голосовой навык подтверждается только при `ack.played_by_client`; сбой TTS после
  `ack.generated` не создаёт `ack.sent_to_media`/`ack.played_by_client` (C-05).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from ..contracts.events import (
    AckGenerated,
    AckPlayedByClient,
    AttemptEvent,
    CallState,
    ComponentName,
    EventSource,
    ModelFailure,
    TextDelivered,
)
from ..faults import ComponentFailure, FaultInjector
from ..ports import MediaTransport, STTProvider, TranscriptSegment, TTSProvider
from .confidence import UncertainEntity, detect_uncertain_entities
from .normalize import normalize_for_tts

_SILENT_STATES = frozenset({CallState.MUTED, CallState.HOLD})


@dataclass(frozen=True)
class SttOutcome:
    """Результат попытки распознать реплику диспетчера."""

    segments: tuple[TranscriptSegment, ...]
    uncertain: tuple[UncertainEntity, ...]
    events: tuple[AttemptEvent, ...]
    text_mode: bool


@dataclass(frozen=True)
class TtsOutcome:
    """Результат попытки озвучить и доставить реплику руководителя."""

    ack_id: UUID
    events: tuple[AttemptEvent, ...]
    confirmed: bool
    text_mode: bool


class VoicePipeline:
    """Единая точка входа голосового тракта для одной попытки/звонка."""

    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        media: MediaTransport,
        faults: FaultInjector,
        *,
        hints: Sequence[str] = (),
        start_seq: int = 0,
        clock: Callable[[], datetime] | None = None,
    ):
        self.stt = stt
        self.tts = tts
        self.media = media
        self.faults = faults
        self.hints = tuple(hints)
        self._seq = start_seq
        self._clock = clock or (lambda: datetime.now(UTC))

    def _next(self, attempt_id: UUID) -> dict:
        self._seq += 1
        return {
            "event_id": uuid4(),
            "attempt_id": attempt_id,
            "seq": self._seq,
            "server_ts": self._clock(),
            "source": EventSource.AI_WORKER,
        }

    def ingest_audio(self, *, call_state: CallState, audio: bytes, attempt_id: UUID) -> SttOutcome:
        """В mute/hold звук ученика не поступает в STT вообще — провайдер не вызывается."""
        if call_state in _SILENT_STATES:
            return SttOutcome((), (), (), text_mode=False)

        try:
            segments = self.faults.call(
                ComponentName.STT,
                lambda: self.stt.transcribe(audio, hints=self.hints),
                attempt_id=attempt_id,
            )
        except ComponentFailure:
            kind = self.faults.log[-1].kind
            event = ModelFailure(**self._next(attempt_id), component=ComponentName.STT, kind=kind)
            return SttOutcome((), (), (event,), text_mode=True)

        segments = tuple(segments)
        uncertain = detect_uncertain_entities(segments)
        return SttOutcome(segments, uncertain, (), text_mode=False)

    def speak_ack(
        self,
        *,
        attempt_id: UUID,
        call_id: UUID,
        utterance_id: UUID,
        text: str,
        voice: str,
        intonation: str,
    ) -> TtsOutcome:
        """`ack.generated` фиксируется до синтеза: сбой TTS после него не даёт доставки (C-05)."""
        ack_id = uuid4()
        events: list[AttemptEvent] = [
            AckGenerated(
                **self._next(attempt_id), ack_id=ack_id, call_id=call_id, utterance_id=utterance_id
            )
        ]
        normalized = normalize_for_tts(text)

        try:
            audio = self.faults.call(
                ComponentName.TTS,
                lambda: self.tts.synthesize(normalized, voice=voice, intonation=intonation),
                attempt_id=attempt_id,
            )
        except ComponentFailure:
            kind = self.faults.log[-1].kind
            events.append(
                ModelFailure(**self._next(attempt_id), component=ComponentName.TTS, kind=kind)
            )
            events.append(TextDelivered(**self._next(attempt_id), utterance_id=utterance_id))
            return TtsOutcome(ack_id, tuple(events), confirmed=False, text_mode=True)

        media_events = self.media.play(
            attempt_id=attempt_id,
            call_id=call_id,
            ack_id=ack_id,
            utterance_id=utterance_id,
            audio=audio,
        )
        events.extend(media_events)
        confirmed = any(isinstance(event, AckPlayedByClient) for event in media_events)
        return TtsOutcome(ack_id, tuple(events), confirmed=confirmed, text_mode=False)
