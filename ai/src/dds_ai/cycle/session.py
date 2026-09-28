"""Сквозной учебный цикл без моделей на моках (M1).

уведомление → карточка → решение → разговор с руководителем → подтверждение → сдача → оценка.

Роли событий: клиент (действия ученика), бэкенд (уведомление, сдача, сценарные состояния),
ИИ-воркер (реплики руководителя, `ack.generated`, сбои моделей), медиатракт (доставка C-05).
Журнал, хранилище карточек и маршрутизация — моки бэкенда (4.3).

Голосовой канал здесь — мок: реплика ученика подаётся готовым текстом (STT — M2), TTS —
через адаптер в интерактивной полосе воркера, воспроизведение — `MockMediaTransport`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from ..contracts.card import CardField, CardRevisionRecord, FieldOrigin, FieldState
from ..contracts.common import ModelRef, VersionRef
from ..contracts.criteria import CriterionResult, RuleStatus
from ..contracts.dialogue import ReplyMode, ScenarioCallState
from ..contracts.difficulty import DifficultyWeights
from ..contracts.events import (
    AckDeliveryUnconfirmed,
    AckGenerated,
    CallState,
    CallStateChanged,
    CardOpened,
    CardRevision,
    ComponentName,
    DispatcherDecision,
    EventSource,
    FailureKind,
    HelpRequested,
    HintShown,
    ModelFailure,
    NotificationShown,
    ScenarioCallStateChanged,
    SelectedAction,
    Speaker,
    SubmitAccepted,
    SubmitClicked,
    TextDelivered,
    Utterance,
)
from ..contracts.mode import AttemptTrack, TrainingMode
from ..contracts.remarks import Remark
from ..contracts.routing import RoutingRequest
from ..contracts.rubric import Rubric
from ..contracts.scenario import Scenario
from ..contracts.scoring import AttemptVersionSnapshot, ScoreSummary
from ..contracts.worker import FailureRecord, Lane
from ..difficulty import load_weights
from ..faults import ComponentFailure
from ..hints.track import HintDenied, attempt_track, can_show_guided_hint
from ..mocks.card_store import InMemoryCardStore
from ..mocks.event_log import InMemoryEventLog, ManualClock
from ..mocks.routing import Rule, TableRoutingEngine
from ..mocks.scoring import score
from ..ports import (
    LLMProvider,
    MediaTransport,
    RoutingEngine,
    SemanticJudge,
    STTProvider,
    TTSProvider,
)
from ..scenarios.card_generation import CARD_SCHEMA, generate_card
from ..supervisor.dialogue import Conversation, DialResult, Supervisor, Turn
from ..supervisor.phrasing import LLMPhraser
from ..worker import InferenceWorker
from .rules import RULES, VOICE_PATH, Channel, Evaluator, Step, final_card_fields, missing_steps

SUPERVISOR_ROLE = "supervisor"
TTS_FAILED = "tts_failed"


def synthetic_routing() -> TableRoutingEngine:
    """Мок таблицы правил для синтетических сценариев (не классификатор ЕКП)."""
    return TableRoutingEngine(
        {
            "fire_residential": Rule(
                "syn-r1", RuleStatus.ACTIVE, "syn-1", "fire_service", ("fire_service", "ambulance")
            ),
            "gas_smell": Rule(
                "syn-r2", RuleStatus.ACTIVE, "syn-2", "gas_service", ("gas_service",)
            ),
            "trash_fire_repeat": Rule("syn-r3", RuleStatus.QUARANTINED, None, None, ()),
        },
        version="synthetic-mock-1",
    )


class IncompleteSubmission(Exception):
    """Сдача с пропущенными шагами требует явного выбора «с невыполненными шагами» (C-02)."""

    def __init__(self, missing: list[Step]):
        super().__init__("missing steps: " + ", ".join(missing))
        self.missing = missing


@dataclass(frozen=True)
class Evaluation:
    results: tuple[CriterionResult, ...]
    summary: ScoreSummary
    remarks: tuple[Remark, ...] = ()
    track: AttemptTrack = AttemptTrack.INDEPENDENT

    def by_id(self, criterion_id: str) -> CriterionResult:
        return next(r for r in self.results if r.criterion_id == criterion_id)


class TrainingSession:
    def __init__(
        self,
        scenario: Scenario,
        *,
        channel: Channel = Channel.TEXT,
        worker: InferenceWorker | None = None,
        routing: RoutingEngine | None = None,
        card_store: InMemoryCardStore | None = None,
        clock: ManualClock | None = None,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
        media: MediaTransport | None = None,
        llm: LLMProvider | None = None,
        judge: SemanticJudge | None = None,
        attempt_id: UUID | None = None,
        dispatcher_id: UUID | None = None,
        card_seed: int | None = None,
        mode: TrainingMode = TrainingMode.INDEPENDENT,
        difficulty_weights: DifficultyWeights | None = None,
    ):
        if channel is Channel.VOICE and (tts is None or media is None):
            raise ValueError("voice channel requires tts and media")
        self.scenario = scenario
        self.channel = channel
        self.mode = mode
        self.attempt_id = attempt_id or uuid4()
        self.dispatcher_id = dispatcher_id or uuid4()
        self.clock = clock or ManualClock()
        self.log = InMemoryEventLog(self.attempt_id, self.clock)
        self.worker = worker or InferenceWorker.from_profile("cpu")
        self.routing = routing or synthetic_routing()
        # G-4: веса сложности фиксируются при старте попытки, не при её оценке/анализе.
        self.difficulty_weights = difficulty_weights or load_weights()
        self.cards = card_store or InMemoryCardStore()
        self.stt, self.tts, self.media, self.judge = stt, tts, media, judge
        if stt is not None:
            self.worker.register(ComponentName.STT, stt)
        if tts is not None:
            self.worker.register(ComponentName.TTS, tts)
        phraser = None
        if llm is not None:
            self.worker.register(ComponentName.LLM, llm)
            phraser = LLMPhraser(llm, self.worker, role_title="руководитель дежурной смены")
        self.supervisor = Supervisor(
            scenario.brief(SUPERVISOR_ROLE),
            scenario=scenario.scenario,
            phraser=phraser,
            attempt_id=self.attempt_id,
        )
        # E-1: карточка — через генератор со сценарной вариацией (6.5, D-003), не фиксированный
        # мок; seed по умолчанию — от attempt_id, поэтому один и тот же attempt воспроизводит
        # ту же карточку без явного seed, а разные попытки естественно получают вариации.
        self.card_seed = card_seed if card_seed is not None else self.attempt_id.int % 2_147_483_647
        self.card = generate_card(scenario, seed=self.card_seed, created_at=self.clock())
        self.cards.put_source(self.card)
        self.conversation: Conversation | None = None
        self.call_id: UUID | None = None
        self.submitted: SubmitAccepted | None = None
        if self.channel is Channel.VOICE:
            self._preflight_voice()
        # G-4: снимок версий фиксируется здесь, при старте попытки, а не при оценке; ни одно
        # из полей ниже не меняется до конца попытки (сценарий, маршрутизация и веса сложности
        # неизменяемы после __init__), поэтому снимок, построенный сейчас, совпадает с тем, что
        # был бы построен в любой другой момент этой же попытки.
        self.difficulty_config = VersionRef(
            name="difficulty-weights", version=self.difficulty_weights.weights_version
        )

    @property
    def events(self):
        return self.log.events

    def version_snapshot(
        self, rubric: Rubric, *, models: tuple[ModelRef, ...] = ()
    ) -> AttemptVersionSnapshot:
        """`AttemptVersionSnapshot` (D-003, C-08, G-4): собран из версий, уже зафиксированных
        при старте попытки — `difficulty_config` несёт версию весов сложности, использованных
        для этой попытки (`self.difficulty_weights`), а не веса, актуальные на момент вызова.

        `classifier` в этом синтетическом цикле не отделён от табличной маршрутизации (M1: нет
        отдельного ML-классификатора) — оба поля ссылаются на версию таблицы правил.
        """
        routing_version = getattr(
            self.routing, "version", VersionRef(name="routing_rules", version="external")
        )
        return AttemptVersionSnapshot(
            attempt_id=self.attempt_id,
            scenario=self.scenario.scenario,
            reference=self.scenario.reference.reference,
            card_schema=CARD_SCHEMA,
            classifier=routing_version,
            routing_rules=routing_version,
            rubric=VersionRef(name="rubric", version=rubric.rubric_version),
            time_policy=RULES,
            difficulty_config=self.difficulty_config,
            models=models,
            created_at=self.clock(),
        )

    @property
    def track(self) -> AttemptTrack:
        """Трек попытки для профиля/рейтинга (D-041): guided остаётся guided; independent
        становится supported, если был запрос помощи, — так же, как решает `hints.attempt_track`."""
        return attempt_track(self.mode, self.log.events)

    # --- подсказки и режимы (6.7, D-041) --------------------------------------------------

    def show_hint(self, text: str, *, hint_id: UUID | None = None) -> None:
        """«Делай как я» — всегда; самостоятельный режим — только после запроса помощи (G-1)."""
        if not can_show_guided_hint(self.mode, self.log.events):
            raise HintDenied("guided hints are only available in guided mode")
        self.log.append(
            HintShown, source=EventSource.AI_WORKER, hint_id=hint_id or uuid4(), text=text
        )

    def request_help(self) -> None:
        """Самостоятельный режим: запрос помощи сразу открывает подсказки (иначе кнопка
        запроса помощи бесполезна — G-1) и переводит попытку в «с поддержкой» (D-041) —
        `track` перестаёт быть `independent`, попытка исключается из самостоятельного
        рейтинга. С этого момента `show_hint()` доступен так же, как в guided.
        """
        if self.mode is not TrainingMode.INDEPENDENT:
            raise RuntimeError("help_requested applies only to the independent mode")
        self.log.append(HelpRequested, source=EventSource.CLIENT)

    # --- действия ученика ----------------------------------------------------------------

    def notify(self) -> None:
        self.log.append(NotificationShown, source=EventSource.BACKEND)

    def open_card(self) -> dict[str, CardField]:
        self.log.append(CardOpened, source=EventSource.CLIENT, card_id=self.card.card_id)
        return {k: f for k, f in self.card.fields.items() if f.visible_to_trainee}

    def edit_card(self, **values: Any) -> None:
        changes = {
            name: CardField(state=FieldState.KNOWN, raw=value, origin=FieldOrigin.DISPATCHER)
            for name, value in values.items()
        }
        revision = len(self.cards.revisions(self.card.card_id)) + 1
        self.cards.add_revision(
            CardRevisionRecord(
                card_id=self.card.card_id,
                revision=revision,
                author_id=self.dispatcher_id,
                created_at=self.clock(),
                changes=changes,
            )
        )
        self.log.append(
            CardRevision,
            source=EventSource.CLIENT,
            card_id=self.card.card_id,
            revision=revision,
            changed_fields=tuple(changes),
        )

    def decide(self, decision: DispatcherDecision) -> None:
        self.log.append(SelectedAction, source=EventSource.CLIENT, decision=decision)

    def dial(self) -> DialResult:
        if self.conversation is not None:
            raise RuntimeError("already in a call")
        call_id = uuid4() if self.channel is Channel.VOICE else None
        result = self.supervisor.dial()
        if result.outcome is not None:
            self._scenario_state(result.outcome, call_id)
            return result
        self.conversation, self.call_id = result.conversation, call_id
        if call_id is not None:
            self.log.append(
                CallStateChanged, source=EventSource.MEDIA, call_id=call_id, state=CallState.ACTIVE
            )
        assert result.greeting is not None
        self._emit(result.greeting)
        return result

    def say(self, text: str) -> Turn:
        if self.conversation is None:
            raise RuntimeError("no active call")
        self.log.append(
            Utterance,
            source=EventSource.CLIENT if self.channel is Channel.TEXT else EventSource.AI_WORKER,
            call_id=self.call_id,
            utterance_id=uuid4(),
            speaker=Speaker.DISPATCHER,
            text=text,
        )
        turn = self.conversation.hear(text)
        if turn.dropped:
            self._scenario_state(ScenarioCallState.DROPPED, self.call_id)
            self._end_call(EventSource.BACKEND)
        else:
            self._emit(turn)
        return turn

    def switch_to_text(self, failure: FailureRecord) -> None:
        """Голосовой тракт не восстановлен — помеченный текстовый режим (C-04).

        Сбой пишется в журнал; активный вызов завершается медиатрактом, а разговор с тем же
        состоянием продолжается текстом. Голосовые критерии получат `technical_error` (M1-3).
        """
        if failure.component not in VOICE_PATH:
            raise ValueError(f"{failure.component} is not a voice path component")
        if self.channel is Channel.TEXT:
            raise RuntimeError("already in text mode")
        self._failure(failure)
        if self.call_id is not None:
            self.log.append(
                CallStateChanged,
                source=EventSource.MEDIA,
                call_id=self.call_id,
                state=CallState.ENDED,
            )
            self.call_id = None
        self.channel = Channel.TEXT

    def hang_up(self) -> None:
        if self.conversation is not None:
            self.conversation.hang_up()
            self._end_call(EventSource.CLIENT)

    def submit(self, *, incomplete: bool = False) -> SubmitAccepted:
        if self.submitted is not None:
            return self.submitted  # повторная сдача не создаёт новых событий
        self.hang_up()
        missing = missing_steps(self.log.events, self.channel)
        if missing and not incomplete:
            raise IncompleteSubmission(missing)
        self.log.append(SubmitClicked, source=EventSource.CLIENT)
        self.submitted = self.log.append(
            SubmitAccepted,
            source=EventSource.BACKEND,
            incomplete=bool(missing),
            missing_steps=tuple(missing),
        )
        return self.submitted

    # --- оценка --------------------------------------------------------------------------

    def evaluate(self, rubric: Rubric) -> Evaluation:
        fields = final_card_fields(self.card.fields, self.cards.revisions(self.card.card_id))
        decision = self.routing.route(RoutingRequest(incident_type=self.scenario.incident_type))
        evaluator = Evaluator(
            self.scenario,
            self.log.events,
            channel=self.channel,
            card_fields=fields,
            routing=decision,
            judge=self.judge,
            injector=self.worker.injector,
            worker=self.worker,
            attempt_id=self.attempt_id,
        )
        results = tuple(evaluator.results(rubric))
        return Evaluation(
            results=results,
            summary=score(rubric, results),
            remarks=evaluator.remarks(),
            track=self.track,
        )

    # --- события ИИ-воркера и медиатракта --------------------------------------------------

    def _scenario_state(self, state: ScenarioCallState, call_id: UUID | None) -> None:
        self.log.append(
            ScenarioCallStateChanged,
            source=EventSource.BACKEND,
            call_id=call_id,
            role_id=SUPERVISOR_ROLE,
            state=state,
        )

    def _preflight_voice(self) -> None:
        """D-3/F-3: провал preflight голосового тракта до старта — сбой пишет воркер (он
        владеет preflight, 6.1), попытка сразу помечается текстовым режимом (C-04); до
        соединения, поэтому без разбора активного звонка. STT входит в preflight только когда
        попытка действительно использует голосовой канал со STT (F-3) — без адаптера STT
        текстовый ввод реплик остаётся мок-путём M1 и preflight его не требует."""
        required = [(ComponentName.TTS, Lane.INTERACTIVE)]
        if self.stt is not None:
            required.append((ComponentName.STT, Lane.INTERACTIVE))
        report = self.worker.preflight(required, voice_path_ok=True)
        if report.ready:
            return
        for f in self.worker.preflight_failures(report, at=self.clock()):
            self._failure(f)
        self.channel = Channel.TEXT

    def _end_call(self, source: EventSource) -> None:
        if self.call_id is not None:
            self.log.append(
                CallStateChanged, source=source, call_id=self.call_id, state=CallState.ENDED
            )
        self.conversation, self.call_id = None, None

    def _failure(self, f: FailureRecord, *, recovered: bool = False) -> None:
        self.log.append(
            ModelFailure,
            source=EventSource.AI_WORKER,
            component=f.component,
            kind=f.kind,
            detail=f.detail,
            recovered=recovered,
        )

    def _emit(self, turn: Turn) -> None:
        reply = turn.reply
        assert reply is not None
        # Отклонённый выход LLM, исправленный повтором, — не сбой (решение D-1).
        llm_ok = reply.mode is ReplyMode.LLM
        for f in turn.failures:
            self._failure(
                f,
                recovered=llm_ok
                and f.component is ComponentName.LLM
                and f.kind is FailureKind.INVALID_OUTPUT,
            )
        utterance = self.log.append(
            Utterance,
            source=EventSource.AI_WORKER,
            call_id=self.call_id,
            utterance_id=uuid4(),
            speaker=Speaker.SUPERVISOR,
            text=reply.text,
            model_ref=reply.model_ref,
            fallback=reply.mode is ReplyMode.FALLBACK,
        )
        if self.channel is Channel.TEXT:
            self.log.append(
                TextDelivered,
                source=EventSource.CLIENT,
                utterance_id=utterance.utterance_id,
                ack_id=turn.ack_id,
            )
            return
        self._speak(utterance, turn.ack_id)

    def _speak(self, utterance: Utterance, ack_id: UUID | None) -> None:
        assert self.call_id is not None and self.tts is not None and self.media is not None
        call_id = self.call_id
        ids = {"call_id": call_id, "utterance_id": utterance.utterance_id}
        if ack_id is not None:
            self.log.append(AckGenerated, source=EventSource.AI_WORKER, ack_id=ack_id, **ids)
        tts = self.tts
        before = len(self.worker.injector.log)
        try:
            audio = self.worker.call(
                ComponentName.TTS,
                Lane.INTERACTIVE,
                lambda: tts.synthesize(utterance.text, voice="default", intonation="neutral"),
                attempt_id=self.attempt_id,
                affected_evidence=(str(utterance.utterance_id),),
            )
        except ComponentFailure:
            for f in self.worker.injector.log[before:]:
                self._failure(f)
            if ack_id is not None:
                self.log.append(
                    AckDeliveryUnconfirmed,
                    source=EventSource.AI_WORKER,
                    ack_id=ack_id,
                    reason=TTS_FAILED,
                    **ids,
                )
            return
        if ack_id is None:
            return  # реплики без подтверждения мок медиатракта не отслеживает
        for event in self.media.play(
            attempt_id=self.attempt_id,
            call_id=call_id,
            ack_id=ack_id,
            utterance_id=utterance.utterance_id,
            audio=audio,
        ):
            self.log.ingest(event)
