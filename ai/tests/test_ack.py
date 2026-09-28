"""C-05: засчитанное подтверждение руководителя — строка матрицы раздела 8 целиком."""

from uuid import uuid4

import pytest

from dds_ai.contracts.criteria import CriterionStatus
from dds_ai.contracts.events import (
    AckDeliveryUnconfirmed,
    AckGenerated,
    AckPlayedByClient,
    AckSentToMedia,
    CallState,
    CallStateChanged,
    ComponentName,
    DispatcherDecision,
    EventSource,
    ModelFailure,
    Speaker,
    TextDelivered,
    Utterance,
)
from dds_ai.contracts.scoring import Verdict
from dds_ai.cycle import Channel, TrainingSession, load_synthetic
from dds_ai.mocks.event_log import InMemoryEventLog
from dds_ai.mocks.media import MockMediaTransport, Playback
from dds_ai.supervisor.ack import HANGUP_REASON, AckOutcome, summarize_acks

from .fakes import FakeTTS, StubJudge

REPORT = (
    "Пожар в жилом доме, ул. Тестовая, д. 12, кв. 5. Дым из окна квартиры. "
    "ДДС реагирует, направляем пожарную охрану. Прошу принять доклад."
)


@pytest.fixture(scope="module")
def fire():
    return load_synthetic()["syn-001-fire-respond"]


def voice_session(fire, *, tts=None, playback=Playback.PLAYED):
    return TrainingSession(
        fire,
        channel=Channel.VOICE,
        tts=tts or FakeTTS(),
        media=MockMediaTransport(playback),
        judge=StubJudge(),
    )


def prepare(s):
    s.notify()
    s.clock.advance(10)
    s.open_card()
    s.edit_card(services=["fire_service", "ambulance"])
    s.decide(DispatcherDecision.RESPOND)


def finish(s, rubric):
    s.clock.advance(60)
    s.submit(incomplete=True)
    return s.evaluate(rubric)


# --- журнал, собранный вручную ---------------------------------------------------------------


class Log:
    def __init__(self):
        self.log = InMemoryEventLog(uuid4())
        self.call_id = uuid4()
        self.utt = uuid4()
        self.ack = uuid4()

    def add(self, cls, source=EventSource.AI_WORKER, **kw):
        return self.log.append(cls, source=source, **kw)

    def call(self, state, source=EventSource.MEDIA, call_id=None):
        return self.add(CallStateChanged, source, call_id=call_id or self.call_id, state=state)

    def ack_event(self, cls, source=EventSource.MEDIA, **kw):
        ids = {"ack_id": self.ack, "call_id": self.call_id, "utterance_id": self.utt}
        return self.add(cls, source, **{**ids, **kw})

    @property
    def events(self):
        return self.log.events


# --- нормальное воспроизведение ----------------------------------------------------------------


def test_normal_playback_gives_exactly_one_ack_for_the_call(fire, rubric):
    s = voice_session(fire)
    prepare(s)
    s.dial()
    call_id = s.call_id
    s.say(REPORT)
    turn = s.say("Да, верно.")
    summary = summarize_acks(s.events)
    assert len(summary.counted) == 1
    rec = summary.counted[0]
    assert rec.ack_id == turn.ack_id and rec.call_id == call_id
    assert summary.counted_for_call(call_id) == 1
    assert summary.counted_for_call(uuid4()) == 0
    types = [e.type for e in s.events if e.type.startswith("ack.")]
    assert types == ["ack.generated", "ack.sent_to_media", "ack.played_by_client"]
    assert len({e.ack_id for e in s.events if e.type.startswith("ack.")}) == 1

    ev = finish(s, rubric)
    assert ev.by_id("voice.ack_received").status is CriterionStatus.PASSED


def test_ack_links_attempt_call_and_supervisor_utterance(fire):
    s = voice_session(fire)
    prepare(s)
    s.dial()
    s.say(REPORT)
    s.say("Да")
    gen = next(e for e in s.events if isinstance(e, AckGenerated))
    utt = next(
        e for e in s.events if isinstance(e, Utterance) and e.utterance_id == gen.utterance_id
    )
    assert utt.speaker is Speaker.SUPERVISOR and gen.attempt_id == s.attempt_id
    assert gen.source is EventSource.AI_WORKER


# --- сброс до подтверждения --------------------------------------------------------------------


def test_hangup_before_confirmation_creates_no_ack(fire, rubric):
    s = voice_session(fire)
    prepare(s)
    s.dial()
    s.say(REPORT)  # руководитель повторил понятое, но подтверждения ещё нет
    s.hang_up()
    assert not any(isinstance(e, AckGenerated) for e in s.events)
    assert summarize_acks(s.events).counted == ()
    ev = finish(s, rubric)
    assert ev.by_id("voice.ack_received").status is CriterionStatus.NOT_DONE


def test_hangup_after_generated_before_playback_is_not_counted():
    lg = Log()
    lg.call(CallState.ACTIVE)
    lg.ack_event(AckGenerated, EventSource.AI_WORKER)
    lg.ack_event(AckSentToMedia)
    lg.call(CallState.ENDED, EventSource.CLIENT)
    lg.ack_event(AckDeliveryUnconfirmed, reason=HANGUP_REASON)
    summary = summarize_acks(lg.events)
    assert summary.counted == ()
    assert summary.records[0].outcome is AckOutcome.HANGUP_BEFORE_PLAYBACK
    assert not summary.records[0].technical


def test_playback_reported_after_call_ended_is_not_counted():
    lg = Log()
    lg.call(CallState.ACTIVE)
    lg.ack_event(AckGenerated, EventSource.AI_WORKER)
    lg.call(CallState.ENDED, EventSource.CLIENT)
    lg.ack_event(AckPlayedByClient)
    summary = summarize_acks(lg.events)
    assert summary.counted == ()
    assert summary.records[0].outcome is AckOutcome.PLAYED_OUTSIDE_CALL


def test_hold_is_not_an_audible_call():
    lg = Log()
    lg.call(CallState.ACTIVE)
    lg.ack_event(AckGenerated, EventSource.AI_WORKER)
    lg.call(CallState.HOLD)
    lg.ack_event(AckPlayedByClient)
    assert summarize_acks(lg.events).counted == ()


def test_muted_dispatcher_still_hears_the_ack():
    lg = Log()
    lg.call(CallState.ACTIVE)
    lg.ack_event(AckGenerated, EventSource.AI_WORKER)
    lg.call(CallState.MUTED)
    lg.ack_event(AckPlayedByClient)
    assert len(summarize_acks(lg.events).counted) == 1


# --- TTS падает после ack.generated ------------------------------------------------------------


def test_tts_failure_after_generated_is_technical_not_passed(fire, rubric):
    tts = FakeTTS()
    s = voice_session(fire, tts=tts)
    prepare(s)
    s.dial()
    s.say(REPORT)
    tts.fail = True
    s.say("Да, верно.")
    types = [e.type for e in s.events][-3:]
    assert types == ["ack.generated", "model_failure", "ack.delivery_unconfirmed"]
    failure = next(e for e in s.events if isinstance(e, ModelFailure))
    assert failure.component is ComponentName.TTS
    unconfirmed = next(e for e in s.events if isinstance(e, AckDeliveryUnconfirmed))
    assert unconfirmed.reason == "tts_failed"
    summary = summarize_acks(s.events)
    assert summary.counted == () and summary.records[0].outcome is AckOutcome.UNCONFIRMED

    ev = finish(s, rubric)
    ack = ev.by_id("voice.ack_received")
    assert ack.status is CriterionStatus.TECHNICAL_ERROR  # не пройден и не ошибка ученика
    assert ev.summary.verdict is Verdict.PROVISIONAL


# --- разрыв после серверного приёма реплики -------------------------------------------------


def test_drop_after_server_received_utterance_creates_no_delivery(fire, rubric):
    s = voice_session(fire, playback=Playback.CALL_DROPPED)
    prepare(s)
    s.dial()
    s.say(REPORT)
    s.say("Да, верно.")  # сервер принял реплику, ответ не доставлен
    received = [e for e in s.events if isinstance(e, Utterance) and e.speaker is Speaker.DISPATCHER]
    assert len(received) == 2
    summary = summarize_acks(s.events)
    assert summary.counted == ()
    assert summary.records[0].outcome is AckOutcome.UNCONFIRMED
    ev = finish(s, rubric)
    assert ev.by_id("voice.ack_received").status is CriterionStatus.TECHNICAL_ERROR


def test_network_drop_without_ack_is_technical_not_student_error(fire, rubric):
    s = voice_session(fire)
    prepare(s)
    s.dial()
    s.say(REPORT)
    s._end_call(EventSource.BACKEND)  # связь оборвалась не по сценарию и не по воле ученика
    ev = finish(s, rubric)
    assert ev.by_id("voice.ack_received").status is CriterionStatus.TECHNICAL_ERROR


# --- повторная доставка того же ack_id -----------------------------------------------------


def test_repeated_delivery_of_same_ack_id_does_not_double(fire, rubric):
    s = voice_session(fire)
    prepare(s)
    s.dial()
    s.say(REPORT)
    s.say("Да")
    played = next(e for e in s.events if isinstance(e, AckPlayedByClient))
    generated = next(e for e in s.events if isinstance(e, AckGenerated))
    single = finish(s, rubric)

    s2_events = [*s.events]
    for _ in range(2):  # повторная доставка и повторное «сформировано»
        s.log.ingest(played)
        s.log.ingest(generated)
    summary = summarize_acks(s.events)
    assert len(summary.counted) == 1
    assert summary.counted[0].repeated_deliveries == 2
    assert len(summarize_acks(s2_events).counted) == 1
    assert s.evaluate(rubric).summary == single.summary


def test_played_for_unknown_ack_is_orphan_and_not_counted():
    lg = Log()
    lg.call(CallState.ACTIVE)
    lg.ack_event(AckPlayedByClient)  # без ack.generated
    summary = summarize_acks(lg.events)
    assert summary.counted == () and len(summary.orphan_played_event_ids) == 1


def test_played_with_other_call_id_is_not_counted():
    lg = Log()
    other = uuid4()
    lg.call(CallState.ACTIVE)
    lg.call(CallState.ACTIVE, call_id=other)
    lg.ack_event(AckGenerated, EventSource.AI_WORKER)
    lg.ack_event(AckPlayedByClient, call_id=other)
    assert summarize_acks(lg.events).counted == ()


# --- «принято» без события ---------------------------------------------------------------------


def test_word_prinyato_without_event_is_not_an_ack():
    lg = Log()
    lg.call(CallState.ACTIVE)
    for speaker, text in ((Speaker.DISPATCHER, "Принято?"), (Speaker.SUPERVISOR, "Принято.")):
        lg.add(
            Utterance,
            call_id=lg.call_id,
            utterance_id=uuid4(),
            speaker=speaker,
            text=text,
        )
    lg.call(CallState.ENDED, EventSource.CLIENT)
    summary = summarize_acks(lg.events)
    assert summary.records == () and summary.counted == ()


def test_dispatcher_saying_prinyato_does_not_confirm(fire, rubric):
    s = voice_session(fire)
    prepare(s)
    s.dial()
    s.say("Принято, принято. Подтверждение получено.")
    assert not any(isinstance(e, AckGenerated) for e in s.events)
    ev = finish(s, rubric)
    assert ev.by_id("voice.ack_received").status is CriterionStatus.NOT_DONE


# --- текстовый режим ------------------------------------------------------------------------------


def test_text_mode_records_text_delivery_not_voice_ack(fire, rubric):
    s = TrainingSession(fire, judge=StubJudge())
    prepare(s)
    s.dial()
    s.say(REPORT)
    turn = s.say("Да")
    assert turn.ack_id is not None
    delivered = [e for e in s.events if isinstance(e, TextDelivered) and e.ack_id]
    assert [e.ack_id for e in delivered] == [turn.ack_id]
    assert not any(e.type.startswith("ack.") for e in s.events)  # голосовых событий нет
    summary = summarize_acks(s.events)
    assert summary.counted == () and summary.text_delivered_ack_ids == (turn.ack_id,)

    ev = finish(s, rubric)
    ack = ev.by_id("voice.ack_received")
    assert ack.status is CriterionStatus.NOT_CHECKED  # не выдаётся за голосовое
    assert "Текстовый режим" in ack.explanation
