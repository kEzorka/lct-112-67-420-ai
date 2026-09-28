"""Голосовой пайплайн через мок медиатракта: mute/hold, отказы STT/TTS, доставка ack (C-04/C-05)."""

from uuid import uuid4

from dds_ai.contracts.events import (
    AckDeliveryUnconfirmed,
    AckGenerated,
    AckPlayedByClient,
    AckSentToMedia,
    CallState,
    ComponentName,
    FailureKind,
    ModelFailure,
    TextDelivered,
)
from dds_ai.faults import FaultInjector
from dds_ai.mocks.media import MockMediaTransport, Playback
from dds_ai.ports import TranscriptSegment
from dds_ai.voice.pipeline import VoicePipeline
from dds_ai.voice.stt.fake import FakeSTTProvider
from dds_ai.voice.tts.fake import FakeTTSProvider

from .conftest import T0


def make_pipeline(
    *, segments=(), playback: Playback = Playback.PLAYED
) -> tuple[VoicePipeline, FakeSTTProvider, FakeTTSProvider, FaultInjector]:
    stt = FakeSTTProvider(segments)
    tts = FakeTTSProvider()
    media = MockMediaTransport(playback, clock=lambda: T0)
    faults = FaultInjector(clock=lambda: T0)
    pipeline = VoicePipeline(stt, tts, media, faults, clock=lambda: T0)
    return pipeline, stt, tts, faults


def test_muted_audio_never_reaches_stt():
    pipeline, stt, _tts, _faults = make_pipeline()
    outcome = pipeline.ingest_audio(call_state=CallState.MUTED, audio=b"pcm", attempt_id=uuid4())
    assert stt.calls == []
    assert outcome.segments == ()
    assert outcome.text_mode is False


def test_hold_audio_never_reaches_stt():
    pipeline, stt, _tts, _faults = make_pipeline()
    outcome = pipeline.ingest_audio(call_state=CallState.HOLD, audio=b"pcm", attempt_id=uuid4())
    assert stt.calls == []
    assert outcome.segments == ()


def test_active_audio_reaches_stt_and_flags_uncertainty():
    distorted = TranscriptSegment(
        text="улица Ленина", start_ms=0, end_ms=500, final=True, confidence=None
    )
    pipeline, stt, _tts, _faults = make_pipeline(segments=[distorted])
    outcome = pipeline.ingest_audio(call_state=CallState.ACTIVE, audio=b"pcm", attempt_id=uuid4())
    assert stt.calls == [b"pcm"]
    assert outcome.segments == (distorted,)
    assert any(e.kind == "address" for e in outcome.uncertain)


def test_stt_failure_gives_text_mode_and_logs_failure():
    pipeline, _stt, _tts, faults = make_pipeline()
    faults.inject(ComponentName.STT, FailureKind.TIMEOUT)
    attempt_id = uuid4()
    outcome = pipeline.ingest_audio(
        call_state=CallState.ACTIVE, audio=b"pcm", attempt_id=attempt_id
    )
    assert outcome.text_mode is True
    assert outcome.segments == ()
    [event] = outcome.events
    assert isinstance(event, ModelFailure)
    assert (event.component, event.kind) == (ComponentName.STT, FailureKind.TIMEOUT)
    assert faults.log[0].kind is FailureKind.TIMEOUT


def test_normal_playback_gives_one_confirmation():
    pipeline, _stt, tts, _faults = make_pipeline()
    attempt_id, call_id, utterance_id = uuid4(), uuid4(), uuid4()
    outcome = pipeline.speak_ack(
        attempt_id=attempt_id,
        call_id=call_id,
        utterance_id=utterance_id,
        text="Информация принята.",
        voice="male",
        intonation="neutral",
    )
    assert outcome.confirmed is True
    assert isinstance(outcome.events[0], AckGenerated)
    assert isinstance(outcome.events[1], AckSentToMedia)
    assert isinstance(outcome.events[2], AckPlayedByClient)
    assert tts.calls[0][0] == "Информация принята."


def test_dropped_call_gives_no_confirmation_but_keeps_ack_generated():
    pipeline, _stt, _tts, _faults = make_pipeline(playback=Playback.CALL_DROPPED)
    outcome = pipeline.speak_ack(
        attempt_id=uuid4(),
        call_id=uuid4(),
        utterance_id=uuid4(),
        text="Принято.",
        voice="male",
        intonation="neutral",
    )
    assert outcome.confirmed is False
    assert isinstance(outcome.events[0], AckGenerated)
    assert isinstance(outcome.events[-1], AckDeliveryUnconfirmed)


def test_tts_failure_after_ack_generated_gives_no_played_by_client():
    pipeline, _stt, _tts, faults = make_pipeline()
    faults.inject(ComponentName.TTS, FailureKind.ERROR)
    outcome = pipeline.speak_ack(
        attempt_id=uuid4(),
        call_id=uuid4(),
        utterance_id=uuid4(),
        text="Принято.",
        voice="male",
        intonation="neutral",
    )
    assert outcome.confirmed is False
    assert outcome.text_mode is True
    assert isinstance(outcome.events[0], AckGenerated)
    assert not any(isinstance(e, (AckSentToMedia, AckPlayedByClient)) for e in outcome.events)
    assert any(
        isinstance(e, ModelFailure) and e.component is ComponentName.TTS for e in outcome.events
    )
    assert any(isinstance(e, TextDelivered) for e in outcome.events)


def test_tts_normalizes_text_before_synthesis():
    pipeline, _stt, tts, _faults = make_pipeline()
    pipeline.speak_ack(
        attempt_id=uuid4(),
        call_id=uuid4(),
        utterance_id=uuid4(),
        text="Высылаем МЧС по адресу ул. Мира, д. 9",
        voice="male",
        intonation="neutral",
    )
    synthesized_text = tts.calls[0][0]
    assert "эм-че-эс" in synthesized_text
    assert "дом девять" in synthesized_text
