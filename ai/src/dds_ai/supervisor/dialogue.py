"""ИИ-руководитель: конечный автомат диалога (6.4, D-026, D-030, C-04).

Автомат выбирает разрешённое действие (выслушать → уточнить → повторить понятое →
подтвердить получение); черновик реплики даёт проверенный шаблон. На M3 LLM, если подключена,
формулирует этот черновик внутри того же действия (`phrasing.py`). Невалидная или недоступная
LLM → шаблонная реплика (C-04). Отказ на инъекцию LLM не формулирует: это фиксированная реплика.

Руководитель получает только `InterlocutorBrief` сценария: роль, свои факты, элементы доклада
и сценарий вызова. Скрытого эталона оценивания у него нет (инвариант 1). Реплика обучаемого —
реплика, а не команда: фразы про оценку, «правильный адрес» и инструкции не меняют состояние
диалога (инвариант 2). На вопрос о неизвестном факте ответ — «неизвестно» (инвариант 3).
Состояние меняется до формулировки, и выход LLM в него не возвращается.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID, uuid4

from ..contracts.card import FieldState
from ..contracts.common import ModelRef, VersionRef
from ..contracts.dialogue import (
    ReplyMode,
    ScenarioCallState,
    SupervisorAction,
    SupervisorReply,
)
from ..contracts.events import Speaker
from ..contracts.scenario import InterlocutorBrief, ReportElement, ScenarioFact
from ..contracts.worker import FailureRecord
from . import matching
from .phrasing import LLMPhraser, PhraseOutcome, validate_reply
from .prompting import FactLine, HistoryLine, PhraseRequest
from .templates import FallbackTemplates

MAX_CLARIFY_PER_ELEMENT = 2

_FACT_STATE_WORDS = {
    FieldState.UNKNOWN: "неизвестно",
    FieldState.NONE: "нет",
    FieldState.NOT_APPLICABLE: "не применимо",
}

_FACT_KEYS = {
    FieldState.KNOWN: "listen.fact_known",
    FieldState.UNKNOWN: "listen.fact_unknown",
    FieldState.NONE: "listen.fact_none",
    FieldState.NOT_APPLICABLE: "listen.fact_not_applicable",
}


class Phase(StrEnum):
    LISTENING = "listening"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    ENDED = "ended"


class TurnKind(StrEnum):
    """Почему руководитель ответил так — для журнала и тестов."""

    GREETING = "greeting"
    REFUSAL = "refusal"
    FACT_ANSWER = "fact_answer"
    CLARIFY = "clarify"
    READ_BACK = "read_back"
    CONFIRM = "confirm"
    AFTER_CONFIRM = "after_confirm"
    DROPPED = "dropped"


@dataclass(frozen=True)
class Turn:
    kind: TurnKind
    reply: SupervisorReply | None
    ack_id: UUID | None = None
    failures: tuple[FailureRecord, ...] = ()
    llm: PhraseOutcome | None = None  # попытки формулировки и замер стадий (M3)

    @property
    def dropped(self) -> bool:
        return self.kind is TurnKind.DROPPED


@dataclass(frozen=True)
class DialResult:
    outcome: ScenarioCallState | None  # None — соединение установлено
    conversation: Conversation | None = None
    greeting: Turn | None = None

    @property
    def connected(self) -> bool:
        return self.conversation is not None


class Supervisor:
    """Собеседник по контракту `InterlocutorBrief`; `role_id` — руководитель или, позже, 112."""

    def __init__(
        self,
        brief: InterlocutorBrief,
        *,
        scenario: VersionRef,
        templates: FallbackTemplates | None = None,
        phraser: LLMPhraser | None = None,
        attempt_id: UUID | None = None,
    ):
        if not brief.enabled:
            raise ValueError(f"interlocutor {brief.role_id} is disabled (D-030)")
        if not brief.report_items:
            raise ValueError(f"interlocutor {brief.role_id} has no report items")
        self.brief = brief
        self.scenario = scenario
        self.templates = templates or FallbackTemplates.load()
        self.phraser = phraser
        self.attempt_id = attempt_id
        self._dials = 0
        self._connections = 0

    @property
    def fallback_ref(self) -> ModelRef:
        """Версия шаблона и сценария — основание каждой шаблонной реплики (инвариант 6)."""
        return ModelRef(
            component="supervisor",
            model_name="fallback-template",
            model_version=self.templates.version,
            prompt_version=f"{self.scenario.name}@{self.scenario.version}",
        )

    def dial(self) -> DialResult:
        """Набор номера: сценарные «занято» / «не отвечает», затем соединение."""
        outcomes = self.brief.dial_outcomes
        self._dials += 1
        if self._dials <= len(outcomes):
            return DialResult(outcome=outcomes[self._dials - 1])
        self._connections += 1
        drop = self.brief.drop_after_turns if self._connections == 1 else None
        conv = Conversation(self, drop_after_turns=drop)
        return DialResult(outcome=None, conversation=conv, greeting=conv.greet())


class Conversation:
    """Один соединённый вызов. Состояние меняется только через `hear()`."""

    def __init__(self, supervisor: Supervisor, *, drop_after_turns: int | None):
        self.sv = supervisor
        self.brief = supervisor.brief
        self.phase = Phase.LISTENING
        self.drop_after_turns = drop_after_turns
        self.turns = 0
        self.coverage: dict[str, str] = {}  # item_id → фраза обучаемого
        self.clarified: dict[ReportElement, int] = {}
        self.ack_id: UUID | None = None
        self._heard: list[str] = []  # реплики обучаемого в этом вызове (контекст для чисел)
        self._history: list[HistoryLine] = []  # история вызова для промпта LLM (данные)
        self._last: str | None = None  # последняя реплика обучаемого

    # --- состояние ------------------------------------------------------------------------

    def snapshot(self) -> tuple:
        """Всё изменяемое состояние диалога — для проверки, что инъекция его не меняет."""
        return (
            self.phase,
            dict(self.coverage),
            dict(self.clarified),
            self.ack_id,
        )

    def missing_elements(self) -> list[ReportElement]:
        out = []
        for element in ReportElement:
            items = [i for i in self.brief.report_items if i.element is element and i.required]
            if any(i.item_id not in self.coverage for i in items):
                out.append(element)
        return out

    # --- ход диалога ------------------------------------------------------------------------

    def greet(self) -> Turn:
        return self._turn(TurnKind.GREETING, SupervisorAction.LISTEN, "listen.greeting")

    def hang_up(self) -> None:
        self.phase = Phase.ENDED

    def hear(self, text: str) -> Turn:
        if self.phase is Phase.ENDED:
            raise RuntimeError("call has ended")
        self.turns += 1
        self._last = text
        if self.drop_after_turns is not None and self.turns >= self.drop_after_turns:
            self.phase = Phase.ENDED  # сценарный обрыв, не технический сбой
            return Turn(kind=TurnKind.DROPPED, reply=None)

        if matching.is_injection(text):
            return self._turn(
                TurnKind.REFUSAL, SupervisorAction.LISTEN, "listen.refusal", llm=False
            )

        self._heard.append(text)
        if self.phase is Phase.CONFIRMED:
            return self._turn(
                TurnKind.AFTER_CONFIRM, SupervisorAction.LISTEN, "listen.after_confirm"
            )

        if self.phase is Phase.AWAITING_CONFIRMATION:
            if matching.is_yes(text) and not matching.is_no(text):
                self.phase = Phase.CONFIRMED
                self.ack_id = uuid4()
                turn = self._turn(
                    TurnKind.CONFIRM, SupervisorAction.CONFIRM_RECEIPT, "confirm_receipt"
                )
                return Turn(turn.kind, turn.reply, self.ack_id, turn.failures, turn.llm)
            if self._absorb(text):
                return self._read_back()
            self.phase = Phase.LISTENING
            return self._turn(TurnKind.CLARIFY, SupervisorAction.CLARIFY, "clarify.correction")

        if matching.is_question(text):
            return self._answer(text)  # вопрос не засчитывается как переданные сведения
        self._absorb(text)
        return self._next_step()

    # --- внутреннее ------------------------------------------------------------------------

    def _absorb(self, text: str) -> bool:
        """Сопоставить реплику с элементами доклада. True — появились новые сведения."""
        parts = matching.clauses(text) or [text]
        changed = False
        for item in self.brief.report_items:
            hit = next((c for c in parts if matching.any_pattern(item.patterns, c)), None)
            if hit is None and matching.any_pattern(item.patterns, text):
                hit = text.strip()
            if hit is not None and self.coverage.get(item.item_id) != hit:
                self.coverage[item.item_id] = hit
                changed = True
        return changed

    def _next_step(self) -> Turn:
        missing = self.missing_elements()
        for element in missing:
            if self.clarified.get(element, 0) < MAX_CLARIFY_PER_ELEMENT:
                self.clarified[element] = self.clarified.get(element, 0) + 1
                return self._turn(TurnKind.CLARIFY, SupervisorAction.CLARIFY, f"clarify.{element}")
        if not self.coverage:  # нечего повторять — продолжаем уточнять суть
            return self._turn(TurnKind.CLARIFY, SupervisorAction.CLARIFY, "clarify.essence")
        return self._read_back()

    def _read_back(self) -> Turn:
        self.phase = Phase.AWAITING_CONFIRMATION
        said: list[str] = []
        for element in ReportElement:
            for item in self.brief.report_items:
                clause = self.coverage.get(item.item_id)
                if item.element is element and clause and clause not in said:
                    said.append(clause)
        return self._turn(
            TurnKind.READ_BACK, SupervisorAction.READ_BACK, "read_back", items="; ".join(said)
        )

    def _answer(self, text: str) -> Turn:
        fact = self._fact_for(text)
        if fact is None:
            return self._turn(
                TurnKind.FACT_ANSWER, SupervisorAction.LISTEN, "listen.unknown_question"
            )
        fields = {"label": fact.label}
        if fact.value is not None:
            fields["value"] = fact.value
        return self._turn(
            TurnKind.FACT_ANSWER,
            SupervisorAction.LISTEN,
            _FACT_KEYS[fact.state],
            used=(fact,),
            **fields,
        )

    def _fact_for(self, text: str) -> ScenarioFact | None:
        for fact in self.brief.facts:
            if fact.question_patterns and matching.any_pattern(fact.question_patterns, text):
                return fact
        return None

    def _turn(
        self,
        kind: TurnKind,
        action: SupervisorAction,
        key: str,
        *,
        used: tuple[ScenarioFact, ...] = (),
        llm: bool = True,
        **fields: str,
    ) -> Turn:
        context = " ".join([*self._heard, *(f.value or "" for f in used)])
        draft = validate_reply(self.sv.templates.render(key, **fields), context)
        failures: tuple[FailureRecord, ...] = ()
        reply = None
        outcome = None
        phraser = self.sv.phraser
        if phraser is not None and llm:
            request = PhraseRequest(
                role_title=phraser.role_title,
                action=action,
                draft=draft,
                facts=tuple(FactLine(f.label, f.value or _FACT_STATE_WORDS[f.state]) for f in used),
                history=tuple(self._history),
                last_utterance=self._last,
            )
            # Разрешённый контекст LLM — черновик и факты хода. Реплики диспетчера сюда не входят:
            # всё, что из них нужно повторить, автомат уже положил в черновик.
            facts = " ".join(f"{f.label} {f.value}" for f in request.facts)
            outcome = phraser.phrase(request, facts, attempt_id=self.sv.attempt_id)
            failures = outcome.failures
            if outcome.text is not None:
                reply = SupervisorReply(
                    action=action,
                    text=outcome.text,
                    used_fact_ids=tuple(f.fact_id for f in used),
                    mode=ReplyMode.LLM,
                    model_ref=phraser.model_ref,
                )
        if reply is None:
            reply = SupervisorReply(
                action=action,
                text=draft,
                used_fact_ids=tuple(f.fact_id for f in used),
                mode=ReplyMode.FALLBACK,
                model_ref=self.sv.fallback_ref,
            )
        reply.check_against(self.brief.interlocutor)
        if self._last is not None:
            self._history.append(HistoryLine(Speaker.DISPATCHER, self._last))
            self._last = None
        self._history.append(HistoryLine(Speaker.SUPERVISOR, reply.text))
        return Turn(kind=kind, reply=reply, failures=failures, llm=outcome)
