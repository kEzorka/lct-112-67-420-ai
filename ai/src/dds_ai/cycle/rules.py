"""Детерминированные проверки критериев M1 по журналу событий (без моделей).

Результат каждого критерия — статус + значение (инвариант 12) с доказательством-событием.
Правило проверяет только то, что можно утверждать без модели; остальное — «не проверено»,
ожидает семантического оценивателя (M5) или эксперта. Технический сбой ≠ ошибка ученика.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any
from uuid import UUID

from ..contracts.card import CardField, FieldOrigin, FieldState
from ..contracts.common import Evidence, EvidenceKind, VersionRef
from ..contracts.criteria import CriterionResult, CriterionStatus, PartialReason
from ..contracts.dialogue import ScenarioCallState
from ..contracts.events import (
    AttemptEvent,
    CallState,
    CallStateChanged,
    CardOpened,
    ComponentName,
    EventSource,
    ModelFailure,
    NotificationShown,
    ScenarioCallStateChanged,
    SelectedAction,
    Speaker,
    SubmitAccepted,
    Utterance,
)
from ..contracts.remarks import Remark
from ..contracts.routing import RoutingDecision
from ..contracts.rubric import Rubric
from ..contracts.scenario import InterlocutorBrief, ReportElement, Scenario
from ..evaluation import judge_criterion
from ..faults import FaultInjector
from ..judging.address_compare import AddressMatch
from ..judging.address_compare import compare as compare_address
from ..ports import SemanticJudge
from ..remarks import build_remarks, grammar_criterion
from ..supervisor import matching
from ..supervisor.ack import summarize_acks

RULES = VersionRef(name="m1-rule-checks", version="1")
OPEN_LIMIT_S = 30  # C-01, равенство допустимо
PROCESSING_LIMIT_S = 180
# manual.grammar — правило по замечаниям локальной проверки (remarks.grammar_criterion), не LLM.
SEMANTIC_CRITERIA = frozenset({"card.circumstances", "manual.additions"})
VOICE_PATH = frozenset({ComponentName.STT, ComponentName.TTS})


class Channel(StrEnum):
    VOICE = "voice"
    TEXT = "text"  # помеченный текстовый режим: голосовой навык не подтверждается (C-04)


class Step(StrEnum):
    """Шаги нормального учебного пути (C-02)."""

    CARD_OPENED = "card_opened"
    DECISION = "selected_action"
    SUPERVISOR_CALL = "supervisor_call"
    SUPERVISOR_ACK = "supervisor_ack"


# --- наблюдения по журналу ----------------------------------------------------------------


def _first(events: Sequence[AttemptEvent], cls: type) -> Any:
    return next((e for e in sorted(events, key=lambda e: e.seq) if isinstance(e, cls)), None)


def dispatcher_utterances(events: Sequence[AttemptEvent]) -> list[Utterance]:
    return [e for e in events if isinstance(e, Utterance) and e.speaker is Speaker.DISPATCHER]


def connected_calls(events: Sequence[AttemptEvent]) -> list[CallStateChanged]:
    return [e for e in events if isinstance(e, CallStateChanged) and e.state is CallState.ACTIVE]


def had_conversation(events: Sequence[AttemptEvent]) -> bool:
    return bool(connected_calls(events) or dispatcher_utterances(events))


def technical_violation(events: Sequence[AttemptEvent]) -> bool:
    """C-01: любой сбой после старта — флаг технического нарушения попытки.

    Не сбой: отклонённый выход LLM, после которого повтор той же модели прошёл (`recovered`).
    """
    return any(isinstance(e, ModelFailure) and not e.recovered for e in events)


def voice_path_failed(events: Sequence[AttemptEvent]) -> bool:
    """В попытке был сбой голосового тракта (STT или TTS)."""
    return any(isinstance(e, ModelFailure) and e.component in VOICE_PATH for e in events)


def missing_steps(events: Sequence[AttemptEvent], channel: Channel) -> list[Step]:
    missing = []
    if _first(events, CardOpened) is None:
        missing.append(Step.CARD_OPENED)
    if _first(events, SelectedAction) is None:
        missing.append(Step.DECISION)
    if not had_conversation(events):
        missing.append(Step.SUPERVISOR_CALL)
    acks = summarize_acks(events)
    delivered = acks.text_delivered_ack_ids if channel is Channel.TEXT else acks.counted
    if not delivered:
        missing.append(Step.SUPERVISOR_ACK)
    return missing


def report_coverage(
    brief: InterlocutorBrief, events: Sequence[AttemptEvent]
) -> tuple[set[str], list[Utterance]]:
    """Какие элементы доклада переданы (правила M1). Инъекции и вопросы не засчитываются."""
    covered: set[str] = set()
    cited: list[Utterance] = []
    for u in dispatcher_utterances(events):
        if matching.is_injection(u.text) or matching.is_question(u.text):
            continue
        hit = {i.item_id for i in brief.report_items if matching.any_pattern(i.patterns, u.text)}
        if hit - covered:
            cited.append(u)
        covered |= hit
    return covered, cited


def final_card_fields(source: dict[str, CardField], revisions: Sequence) -> dict[str, CardField]:
    fields = dict(source)
    for rev in revisions:
        fields.update(rev.changes)
    return fields


def _norm(value: Any) -> str:
    text = matching.normalize(str(value))
    return " ".join(matching.tokens(text))


def _norm_set(value: Any) -> frozenset[str]:
    items = value if isinstance(value, list | tuple | set | frozenset) else str(value).split(",")
    return frozenset(_norm(v) for v in items if _norm(v))


# --- построение результатов ------------------------------------------------------------------


def _ev(event: AttemptEvent, excerpt: str | None = None) -> Evidence:
    return Evidence(kind=EvidenceKind.EVENT, ref=str(event.event_id), excerpt=excerpt)


def _verified(
    cid: str,
    status: CriterionStatus,
    evidence: Sequence[Evidence],
    explanation: str,
    *,
    value: float | None = None,
    partial: PartialReason | None = None,
) -> CriterionResult:
    if value is None:
        value = 1 if status is CriterionStatus.PASSED else 0
    return CriterionResult(
        criterion_id=cid,
        status=status,
        value=value,
        partial_reason=partial,
        evidence=tuple(evidence),
        rule_ref=RULES,
        explanation=explanation,
    )


def _unverified(cid: str, status: CriterionStatus, explanation: str, **kw: Any) -> CriterionResult:
    return CriterionResult(criterion_id=cid, status=status, explanation=explanation, **kw)


class Evaluator:
    def __init__(
        self,
        scenario: Scenario,
        events: Sequence[AttemptEvent],
        *,
        channel: Channel,
        card_fields: dict[str, CardField],
        routing: RoutingDecision,
        judge: SemanticJudge | None = None,
        injector: FaultInjector | None = None,
        worker: Any = None,
        attempt_id: UUID | None = None,
    ):
        self.s = scenario
        self.events = sorted(events, key=lambda e: e.seq)
        self.channel = channel
        self.card = card_fields
        self.routing = routing
        self.judge = judge
        self.injector = injector or FaultInjector()
        self.worker = worker
        self.attempt_id = attempt_id
        self.submit = _first(self.events, SubmitAccepted)
        if self.submit is None:
            raise ValueError("attempt is not submitted: nothing to evaluate")
        self.technical = technical_violation(self.events)

    def results(self, rubric: Rubric) -> list[CriterionResult]:
        out = []
        for g in rubric.groups:
            for c in g.criteria:
                out.append(self.criterion(c.criterion_id))
        return out

    def remarks(self) -> tuple[Remark, ...]:
        """Замечания попытки (D-037): грамматика ручного ввода, технические сбои,
        неопределённость источника маршрутизации."""
        return build_remarks(
            card_fields=self.card,
            events=self.events,
            routing=self.routing,
            attempt_id=self.submit.attempt_id,
        )

    def criterion(self, cid: str) -> CriterionResult:
        if cid in self.s.reference.not_applicable_criteria:
            return _unverified(
                cid, CriterionStatus.NOT_APPLICABLE, "Неприменимо по эталону сценария до старта."
            )
        if cid in SEMANTIC_CRITERIA:
            return self._semantic(cid)
        check = {
            "time.open": self._time_open,
            "time.processing": self._time_processing,
            "process.sequence": self._sequence,
            "routing.decision": self._decision,
            "routing.services": self._services,
            "card.address": self._address,
            "card.incident_type": self._incident_type,
            "voice.call_made": self._call_made,
            "voice.facts_transferred": self._facts_transferred,
            "voice.ack_received": self._ack_received,
            "manual.grammar": self._grammar,
        }.get(cid)
        if check is None:
            return _unverified(cid, CriterionStatus.NOT_CHECKED, "Для критерия нет правила M1.")
        return check(cid)

    # --- время (C-01) ---------------------------------------------------------------------

    def _time_open(self, cid: str) -> CriterionResult:
        if self.technical:
            return _unverified(
                cid,
                CriterionStatus.TECHNICAL_ERROR,
                "Технический сбой в попытке: нормативный временной вердикт не выносится (C-01).",
            )
        shown, opened = _first(self.events, NotificationShown), _first(self.events, CardOpened)
        if opened is None or shown is None:
            return _verified(
                cid,
                CriterionStatus.NOT_DONE,
                [_ev(self.submit, "карточка не открыта до сдачи")],
                "Карточка не открыта.",
            )
        dur = (opened.server_ts - shown.server_ts).total_seconds()
        evidence = [_ev(shown), _ev(opened)]
        if dur <= OPEN_LIMIT_S:
            return _verified(
                cid, CriterionStatus.PASSED, evidence, f"Открыта за {dur:g} с (≤ {OPEN_LIMIT_S})."
            )
        return _verified(
            cid,
            CriterionStatus.PASSED,
            evidence,
            f"Открыта за {dur:g} с — позже норматива {OPEN_LIMIT_S} с (W-01, 0,5).",
            value=0.5,
            partial=PartialReason.LATE,
        )

    def _time_processing(self, cid: str) -> CriterionResult:
        if self.technical:
            return _unverified(
                cid,
                CriterionStatus.TECHNICAL_ERROR,
                "Технический сбой в попытке: нормативный временной вердикт не выносится (C-01).",
            )
        opened = _first(self.events, CardOpened)
        if opened is None:
            return _verified(
                cid,
                CriterionStatus.NOT_DONE,
                [_ev(self.submit, "карточка не открыта до сдачи")],
                "Обработка не начата: карточка не открыта.",
            )
        dur = (self.submit.server_ts - opened.server_ts).total_seconds()
        evidence = [_ev(opened), _ev(self.submit)]
        if dur <= PROCESSING_LIMIT_S:
            return _verified(
                cid,
                CriterionStatus.PASSED,
                evidence,
                f"Обработка {dur:g} с (≤ {PROCESSING_LIMIT_S}).",
            )
        return _verified(
            cid,
            CriterionStatus.FAILED,
            evidence,
            f"Обработка {dur:g} с > {PROCESSING_LIMIT_S} с — критическая ошибка W-01.",
        )

    def _sequence(self, cid: str) -> CriterionResult:
        missing = [
            s
            for s in missing_steps(self.events, self.channel)
            if s is not Step.SUPERVISOR_ACK  # подтверждение оценивает voice.ack_received
        ]
        if missing:
            return _verified(
                cid,
                CriterionStatus.NOT_DONE,
                [_ev(self.submit, "пропущены: " + ", ".join(missing))],
                "Пропущены шаги: " + ", ".join(missing) + " (C-02).",
            )
        steps = [
            _first(self.events, NotificationShown),
            _first(self.events, CardOpened),
            _first(self.events, SelectedAction),
            (dispatcher_utterances(self.events) or connected_calls(self.events))[0],
            self.submit,
        ]
        steps = [s for s in steps if s is not None]
        evidence = [_ev(s) for s in steps]
        if [s.seq for s in steps] == sorted(s.seq for s in steps):
            return _verified(
                cid,
                CriterionStatus.PASSED,
                evidence,
                "Уведомление → карточка → решение → доклад → сдача.",
            )
        return _verified(cid, CriterionStatus.FAILED, evidence, "Нарушен порядок действий.")

    # --- решение и маршрутизация ------------------------------------------------------------

    def _decision(self, cid: str) -> CriterionResult:
        chosen = [e for e in self.events if isinstance(e, SelectedAction)]
        if not chosen:
            return _verified(
                cid,
                CriterionStatus.NOT_DONE,
                [_ev(self.submit, "решение не выбрано до сдачи")],
                "Решение не выбрано.",
            )
        last = chosen[-1]
        expected = self.s.reference.expected_decision
        if last.decision is expected:
            return _verified(cid, CriterionStatus.PASSED, [_ev(last)], "Решение верное.")
        return _verified(
            cid,
            CriterionStatus.FAILED,
            [_ev(last)],
            f"Выбрано «{last.decision}», по сценарию требуется другое решение.",
        )

    def _services(self, cid: str) -> CriterionResult:
        rd = self.routing
        if not rd.auto_gradable:
            return _unverified(
                cid,
                CriterionStatus.NOT_CHECKED,
                f"Правило маршрутизации {rd.rule_id or '—'} не активно: решает эксперт.",
                rule_ref=rd.routing_rules,
                rule_status=rd.rule_status,
            )
        field = self.card.get("services")
        rule_ev = Evidence(kind=EvidenceKind.RULE, ref=f"{rd.routing_rules.version}:{rd.rule_id}")
        if field is None or field.state is not FieldState.KNOWN:
            return _verified(
                cid,
                CriterionStatus.NOT_DONE,
                [rule_ev, _ev(self.submit, "службы не указаны до сдачи")],
                "Службы не указаны в карточке.",
            )
        chosen, required = _norm_set(field.raw), _norm_set(rd.services)
        card_ev = Evidence(kind=EvidenceKind.CARD_FIELD, ref="services", excerpt=str(field.raw))
        if chosen == required:
            return _verified(
                cid, CriterionStatus.PASSED, [rule_ev, card_ev], "Службы совпадают с правилом."
            )
        return _verified(
            cid,
            CriterionStatus.FAILED,
            [rule_ev, card_ev],
            "Состав служб не совпадает с правилом маршрутизации.",
        )

    # --- карточка --------------------------------------------------------------------------

    def _field(self, name: str) -> tuple[CardField | None, str | None]:
        return self.card.get(name), self.s.reference.expected_fields.get(name)

    def _address(self, cid: str) -> CriterionResult:
        """Точное сравнение с нормализацией (D-034); при расхождении — структурное сравнение
        по компонентам (M5, `judging.address_compare`), не строкой целиком: неполный адрес
        (без дома/квартиры) отличается от неверного (другой населённый пункт/улица/дом)."""
        field, expected = self._field("address")
        if field is None or expected is None or field.state is not FieldState.KNOWN:
            return _unverified(cid, CriterionStatus.NOT_CHECKED, "Нет данных для сравнения адреса.")
        ev = Evidence(kind=EvidenceKind.CARD_FIELD, ref="address", excerpt=str(field.raw))
        if _norm(field.raw) == _norm(expected):
            return _verified(cid, CriterionStatus.PASSED, [ev], "Адрес совпадает.")
        cmp = compare_address(str(field.raw), expected)
        if cmp.match is AddressMatch.EXACT:
            return _verified(cid, CriterionStatus.PASSED, [ev], "Адрес совпадает по структуре.")
        if cmp.match is AddressMatch.INCOMPLETE:
            return _verified(
                cid,
                CriterionStatus.PASSED,
                [ev],
                f"Адрес неполный: не указано «{cmp.labels(cmp.missing_groups)}» — остальное "
                "совпадает с эталоном (W-01, 0,5).",
                value=0.5,
                partial=PartialReason.INACCURATE_FIELD,
            )
        if cmp.match is AddressMatch.WRONG:
            return _verified(
                cid,
                CriterionStatus.FAILED,
                [ev],
                f"Адрес неверный: не совпадает «{cmp.labels(cmp.differing_groups)}» — "
                "критическая ошибка W-01.",
            )
        return _unverified(
            cid,
            CriterionStatus.NOT_CHECKED,
            "Не удалось разобрать структуру адреса для сравнения — решает эксперт.",
        )

    def _incident_type(self, cid: str) -> CriterionResult:
        field, expected = self._field("incident_type")
        if field is None or expected is None or field.state is not FieldState.KNOWN:
            return _unverified(cid, CriterionStatus.NOT_CHECKED, "Нет данных о типе происшествия.")
        ev = Evidence(kind=EvidenceKind.CARD_FIELD, ref="incident_type", excerpt=str(field.raw))
        if _norm(field.raw) == _norm(expected):
            return _verified(cid, CriterionStatus.PASSED, [ev], "Тип происшествия верный.")
        return _verified(cid, CriterionStatus.FAILED, [ev], "Тип происшествия неверный.")

    def _dispatcher_text(self, name: str) -> str | None:
        field = self.card.get(name)
        if field is None or field.origin is not FieldOrigin.DISPATCHER:
            return None
        if field.state is not FieldState.KNOWN or not isinstance(field.raw, str):
            return None
        return field.raw

    def _fact_options(self, *, exclude_topics: frozenset[str]) -> list[dict]:
        return [
            {"label": f.label, "values": [f.value, *f.variant_values]}
            for f in self.s.published_facts
            if f.topic not in exclude_topics and f.state is FieldState.KNOWN
        ]

    def _semantic(self, cid: str) -> CriterionResult:
        """D-034: структурированные поля — точное сравнение (правило); свободный текст —
        семантически по фактам эталона (M5, `judging.LLMSemanticJudge`). Проверяются только
        поля, которые фактически заполнил/изменил диспетчер — исходный дефект карточки от
        ИИ-оператора 112 не штрафуется (не заданное диспетчеру исправление)."""
        if self.judge is None:
            return _unverified(
                cid,
                CriterionStatus.NOT_CHECKED,
                "Семантический оцениватель не подключён (M5): ожидает эксперта.",
            )
        dedicated = frozenset({"address", "incident_type", "services"})
        if cid == "card.circumstances":
            field_name = "description"
            student_text = self._dispatcher_text(field_name)
            exclude = dedicated
        elif cid == "manual.additions":
            others = {
                name: f.raw
                for name, f in self.card.items()
                if f.origin is FieldOrigin.DISPATCHER
                and f.state is FieldState.KNOWN
                and isinstance(f.raw, str)
                and name not in dedicated | {"description"}
            }
            if others:
                field_name, student_text = sorted(others.items())[0]
            else:
                field_name, student_text = "manual", None
            exclude = dedicated | {"description"}
        else:
            return _unverified(
                cid, CriterionStatus.NOT_CHECKED, "Нет контекста для семантической проверки."
            )
        context = {
            "kind": "field_fact",
            "field_name": field_name,
            "student_text": student_text,
            "expected_facts": self._fact_options(exclude_topics=exclude),
        }
        return judge_criterion(
            self.judge,
            self.injector,
            cid,
            context,
            attempt_id=self.attempt_id,
            worker=self.worker,
        )

    def _grammar(self, cid: str) -> CriterionResult:
        """manual.grammar — правило по локальной проверке грамотности (`grammar.checker`), не
        LLM: замечание содержит фрагмент и исправление, текст ученика не меняется (инв. 10)."""
        return grammar_criterion(self.card, self.remarks())

    # --- разговор с руководителем (D-036, C-05) ---------------------------------------------

    def _voice_precheck(self, cid: str) -> CriterionResult | None:
        if not had_conversation(self.events):
            return _verified(
                cid,
                CriterionStatus.NOT_DONE,
                [_ev(self.submit, "звонка руководителю не было до сдачи")],
                "Звонка руководителю не было (C-02).",
            )
        if self.channel is Channel.TEXT:
            if voice_path_failed(self.events):  # решение оркестратора M1-3
                return _unverified(
                    cid,
                    CriterionStatus.TECHNICAL_ERROR,
                    "Текстовый режим включён из-за сбоя голосового тракта: голосовой навык "
                    "не подтверждается по технической причине (C-04, C-01).",
                )
            return _unverified(
                cid,
                CriterionStatus.NOT_CHECKED,
                "Текстовый режим: разговор состоялся текстом, голосовой навык не "
                "подтверждается (C-04, C-05).",
            )
        return None

    def _call_made(self, cid: str) -> CriterionResult:
        pre = self._voice_precheck(cid)
        if pre is not None:
            return pre
        calls = connected_calls(self.events)
        if not calls:
            return _verified(
                cid,
                CriterionStatus.NOT_DONE,
                [_ev(self.submit, "соединения не было")],
                "Соединения с руководителем не было.",
            )
        return _verified(cid, CriterionStatus.PASSED, [_ev(calls[0])], "Звонок руководителю.")

    def _facts_transferred(self, cid: str) -> CriterionResult:
        """D-036: сначала быстрое совпадение по ключевым словам (M1); то, что оно не находит,
        — не штраф, а попытка семантической проверки (M5, паравраза/лёгкое повреждение
        транскрипта учитываются). Она никогда не понижает до `failed` — только подтверждает
        (`passed`) или оставляет «не проверено» (устойчивость к STT, раздел 8 промпта)."""
        pre = self._voice_precheck(cid)
        if pre is not None:
            return pre
        brief = self.s.brief("supervisor")
        covered, cited = report_coverage(brief, self.events)
        utterances = dispatcher_utterances(self.events)
        if not utterances:
            return _verified(
                cid,
                CriterionStatus.NOT_DONE,
                [_ev(self.submit, "доклада не было")],
                "Доклад руководителю не передан.",
            )
        missing_items = [i for i in brief.report_items if i.required and i.item_id not in covered]
        if not missing_items:
            return _verified(
                cid,
                CriterionStatus.PASSED,
                [_ev(u, u.text) for u in cited],
                "Переданы суть, место, обстоятельства, решение ДДС и запрос.",
            )
        missing_labels = sorted(
            {i.element.value for i in missing_items},
            key=lambda e: list(ReportElement).index(ReportElement(e)),
        )
        if self.judge is not None:
            context = {
                "kind": "conversation_coverage",
                "transcript": [
                    {"event_id": str(u.event_id), "text": u.text}
                    for u in utterances
                    if not matching.is_injection(u.text)
                ],
                "items": [
                    {"item_id": i.item_id, "description": " / ".join(i.patterns)}
                    for i in missing_items
                ],
            }
            semantic = judge_criterion(
                self.judge,
                self.injector,
                cid,
                context,
                attempt_id=self.attempt_id,
                worker=self.worker,
            )
            if semantic.status is CriterionStatus.PASSED:
                return _verified(
                    cid,
                    CriterionStatus.PASSED,
                    [_ev(u, u.text) for u in cited] + list(semantic.evidence),
                    "Переданы суть, место, обстоятельства, решение ДДС и запрос (часть — "
                    "семантическая проверка, перефразирование учтено, M5).",
                )
        return _unverified(
            cid,
            CriterionStatus.NOT_CHECKED,
            "Не найдены по ключевым словам M1 и не подтверждены семантически: "
            + ", ".join(missing_labels)
            + ". Ожидает эксперта.",
        )

    def _ack_received(self, cid: str) -> CriterionResult:
        pre = self._voice_precheck(cid)
        if pre is not None:
            return pre
        acks = summarize_acks(self.events)
        if acks.counted:
            rec = acks.counted[0]
            played = next(e for e in self.events if e.event_id == rec.played_event_id)
            return _verified(
                cid,
                CriterionStatus.PASSED,
                [_ev(played)],
                "Подтверждение воспроизведено клиентом в активном вызове (C-05).",
            )
        if acks.technical or self.technical or self._technical_call_end():
            return _unverified(
                cid,
                CriterionStatus.TECHNICAL_ERROR,
                "Доставка подтверждения не подтверждена по технической причине (C-05).",
            )
        return _verified(
            cid,
            CriterionStatus.NOT_DONE,
            [_ev(self.submit, "подтверждение не получено до сдачи")],
            "Подтверждение руководителя не получено: разговор завершён до подтверждения.",
        )

    def _technical_call_end(self) -> bool:
        dropped_by_scenario = False
        for e in self.events:
            if isinstance(e, ScenarioCallStateChanged) and e.state is ScenarioCallState.DROPPED:
                dropped_by_scenario = True
            elif isinstance(e, CallStateChanged) and e.state is CallState.ENDED:
                if e.source is not EventSource.CLIENT and not dropped_by_scenario:
                    return True
                dropped_by_scenario = False
        return False
