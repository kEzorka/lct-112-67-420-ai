"""M5: семантический оцениватель, грамматика, структурное сравнение адреса, замечания,
генерация карточки в цикле (E-1), сбой preflight голосового тракта (D-3).

Матрица раздела 8 промпта: Rule status уже покрыт (test_criteria.py); здесь — эквивалентность
перефразирований, STT-устойчивость, невалидный выход оценивателя, недоступное поле,
неизменность текста при проверке грамматики, экспертная поправка -> новая `ScoreVersion`.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from dds_ai.contracts.common import ModelRef, VersionRef
from dds_ai.contracts.criteria import CriterionStatus
from dds_ai.contracts.events import ComponentName, DispatcherDecision, FailureKind, ModelFailure
from dds_ai.contracts.remarks import RemarkType
from dds_ai.contracts.routing import RoutingDecision
from dds_ai.contracts.scoring import ScoreVersion
from dds_ai.cycle import Channel, TrainingSession, load_synthetic
from dds_ai.faults import FaultInjector, InvalidOutput
from dds_ai.grammar.checker import IssueKind, check_text
from dds_ai.judging.address_compare import AddressMatch
from dds_ai.judging.address_compare import compare as compare_address
from dds_ai.judging.policy import JudgePolicy
from dds_ai.judging.semantic_judge import LLMSemanticJudge
from dds_ai.remarks import build_remarks, grammar_criterion
from dds_ai.scenarios.card_generation import generate_card
from dds_ai.worker import InferenceWorker

from .fakes import FakeSTT, FakeTTS

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
FIRE_REPORT = (
    "Пожар в жилом доме, ул. Тестовая, д. 12, кв. 5. Дым из окна квартиры. "
    "ДДС реагирует, направляем пожарную охрану. Прошу принять доклад."
)


@pytest.fixture(scope="module")
def scenarios():
    return load_synthetic()


@pytest.fixture
def fire(scenarios):
    return scenarios["syn-001-fire-respond"]


# --- E-1: генерация карточки в цикле -------------------------------------------------------


def test_session_uses_generated_card_not_fixed_mock(fire):
    s = TrainingSession(fire, attempt_id=uuid4())
    assert s.card.generation.scenario == fire.scenario
    assert s.card.generation.seed == s.card_seed
    assert all(f.origin == "operator_112" for f in s.card.fields.values())


def test_same_attempt_id_reproduces_same_card(fire):
    aid = uuid4()
    a = TrainingSession(fire, attempt_id=aid)
    b = TrainingSession(fire, attempt_id=aid)
    assert a.card.generation.variation_params == b.card.generation.variation_params
    assert {k: v.raw for k, v in a.card.fields.items()} == {
        k: v.raw for k, v in b.card.fields.items()
    }


def test_explicit_seed_overrides_attempt_derived_seed(fire):
    s1 = TrainingSession(fire, attempt_id=uuid4(), card_seed=7)
    s2 = TrainingSession(fire, attempt_id=uuid4(), card_seed=7)
    assert s1.card_seed == s2.card_seed == 7
    assert generate_card(fire, seed=7, created_at=T0).fields.keys() == s1.card.fields.keys()


# --- D-3: сбой preflight голосового тракта пишет воркер ------------------------------------


def test_voice_preflight_failure_writes_model_failure_and_switches_text(fire):
    worker = InferenceWorker.from_profile("cpu")
    worker.injector.inject(ComponentName.TTS, FailureKind.ERROR)
    from dds_ai.mocks.media import MockMediaTransport

    s = TrainingSession(
        fire,
        channel=Channel.VOICE,
        tts=FakeTTS(),
        media=MockMediaTransport(),
        worker=worker,
    )
    assert s.channel is Channel.TEXT  # C-04: голосовой тракт не подтверждён — текстовый режим
    failures = [e for e in s.events if isinstance(e, ModelFailure)]
    assert failures and failures[0].component == ComponentName.TTS
    assert failures[0].source == "ai_worker"  # решение D-3: пишет воркер, не клиент


def test_voice_preflight_ok_keeps_voice_channel(fire):
    from dds_ai.mocks.media import MockMediaTransport

    s = TrainingSession(fire, channel=Channel.VOICE, tts=FakeTTS(), media=MockMediaTransport())
    assert s.channel is Channel.VOICE
    assert not any(isinstance(e, ModelFailure) for e in s.events)


# --- F-3: STT входит в preflight, когда попытка использует голосовой канал со STT ----------


def test_stt_preflight_failure_writes_model_failure_and_switches_text(fire):
    worker = InferenceWorker.from_profile("cpu")
    worker.injector.inject(ComponentName.STT, FailureKind.ERROR)
    from dds_ai.mocks.media import MockMediaTransport

    s = TrainingSession(
        fire,
        channel=Channel.VOICE,
        stt=FakeSTT(),
        tts=FakeTTS(),
        media=MockMediaTransport(),
        worker=worker,
    )
    assert s.channel is Channel.TEXT  # F-3/C-04: голосовой тракт не подтверждён без STT
    failures = [e for e in s.events if isinstance(e, ModelFailure)]
    stt_failures = [f for f in failures if f.component == ComponentName.STT]
    assert stt_failures
    assert stt_failures[0].source == "ai_worker"  # D-3: пишет воркер, не клиент


def test_stt_preflight_ok_keeps_voice_channel(fire):
    from dds_ai.mocks.media import MockMediaTransport

    s = TrainingSession(
        fire, channel=Channel.VOICE, stt=FakeSTT(), tts=FakeTTS(), media=MockMediaTransport()
    )
    assert s.channel is Channel.VOICE
    assert not any(isinstance(e, ModelFailure) for e in s.events)


def test_voice_without_stt_adapter_does_not_require_stt_in_preflight(fire):
    """F-3: preflight требует STT только когда попытка действительно его использует — без
    адаптера STT (мок-путь M1: реплики подаются готовым текстом) его отказ не должен мешать
    попытке, которая STT не вызывает."""
    worker = InferenceWorker.from_profile("cpu")
    worker.injector.inject(ComponentName.STT, FailureKind.ERROR)
    from dds_ai.mocks.media import MockMediaTransport

    s = TrainingSession(
        fire, channel=Channel.VOICE, tts=FakeTTS(), media=MockMediaTransport(), worker=worker
    )
    assert s.channel is Channel.VOICE
    assert not any(isinstance(e, ModelFailure) for e in s.events)


def test_stt_preflight_failure_makes_voice_criteria_technical_error(fire, rubric):
    """F-3: провал STT preflight -> голосовые критерии technical_error (C-04), не штраф."""
    worker = InferenceWorker.from_profile("cpu")
    worker.injector.inject(ComponentName.STT, FailureKind.ERROR)
    from dds_ai.mocks.media import MockMediaTransport

    s = TrainingSession(
        fire,
        channel=Channel.VOICE,
        stt=FakeSTT(),
        tts=FakeTTS(),
        media=MockMediaTransport(),
        worker=worker,
    )
    assert s.channel is Channel.TEXT
    s.notify()
    s.clock.advance(20)
    s.open_card()
    s.edit_card(services=["fire_service", "ambulance"], description="Дым из окна квартиры.")
    s.decide(DispatcherDecision.RESPOND)
    s.dial()
    s.say(FIRE_REPORT)
    s.say("Да, верно.")
    s.clock.advance(170)
    s.submit()
    evaluation = s.evaluate(rubric)
    voice_results = [r for r in evaluation.results if r.criterion_id.startswith("voice.")]
    assert voice_results
    assert all(r.status is CriterionStatus.TECHNICAL_ERROR for r in voice_results)


# --- структурное сравнение адреса -----------------------------------------------------------

FULL_ADDRESS = "г. Учебный, ул. Тестовая, д. 12, кв. 5"


def test_address_compare_exact():
    assert compare_address(FULL_ADDRESS, FULL_ADDRESS).match is AddressMatch.EXACT


def test_address_compare_incomplete_missing_house_and_flat():
    cmp = compare_address("г. Учебный, ул. Тестовая", FULL_ADDRESS)
    assert cmp.match is AddressMatch.INCOMPLETE
    assert "house" in cmp.missing_groups and "flat" in cmp.missing_groups


def test_address_compare_wrong_house_is_critical():
    cmp = compare_address("г. Учебный, ул. Тестовая, д. 14, кв. 5", FULL_ADDRESS)
    assert cmp.match is AddressMatch.WRONG
    assert "house" in cmp.differing_groups


def test_address_compare_wrong_street_is_critical():
    cmp = compare_address("г. Учебный, ул. Лесная, д. 12, кв. 5", FULL_ADDRESS)
    assert cmp.match is AddressMatch.WRONG
    assert "street" in cmp.differing_groups


def test_address_compare_unparseable_defers_to_expert():
    assert compare_address("тут", FULL_ADDRESS).match is AddressMatch.UNPARSEABLE


def test_address_criterion_in_cycle_gives_concrete_explanation(fire, rubric):
    s = TrainingSession(fire, attempt_id=uuid4())
    s.notify()
    s.clock.advance(10)
    s.open_card()
    s.edit_card(address="г. Учебный, ул. Тестовая, д. 14, кв. 5", services=["fire_service"])
    s.decide(DispatcherDecision.RESPOND)
    s.clock.advance(60)
    s.submit(incomplete=True)
    ev = s.evaluate(rubric)
    r = ev.by_id("card.address")
    assert r.status is CriterionStatus.FAILED and r.value == 0
    assert "дом" in r.explanation


# --- грамматика: не меняет текст ученика, влияет только на manual.grammar -------------------


def test_grammar_checker_finds_issues_without_mutating_text():
    text = "Дым дым из окна ,квартиры.потом ещё что-то"
    issues = check_text(text)
    kinds = {i.kind for i in issues}
    assert IssueKind.REPEATED_WORD in kinds
    assert IssueKind.SPACE_BEFORE_PUNCT in kinds or IssueKind.NO_SPACE_AFTER_PUNCT in kinds
    assert text == "Дым дым из окна ,квартиры.потом ещё что-то"  # инвариант 10


def test_grammar_criterion_not_checked_when_no_manual_text():
    result = grammar_criterion({}, remarks=())
    assert result.status is CriterionStatus.NOT_CHECKED  # недоступное поле — не штраф


def test_grammar_criterion_scores_by_remark_count(fire):
    from dds_ai.contracts.card import CardField, FieldOrigin, FieldState

    clean_fields = {
        "description": CardField(
            state=FieldState.KNOWN, raw="Дым из окна квартиры.", origin=FieldOrigin.DISPATCHER
        )
    }
    remarks = build_remarks(
        card_fields=clean_fields,
        events=(),
        routing=RoutingDecision(
            routing_rules=VersionRef(name="r", version="1"), rule_id=None, rule_status=None
        ),
        attempt_id=uuid4(),
    )
    ok = grammar_criterion(clean_fields, remarks)
    assert ok.status is CriterionStatus.PASSED and ok.value == 1

    bad_fields = {
        "description": CardField(
            state=FieldState.KNOWN,
            raw="Дым дым из окна квартиры дым дым.",
            origin=FieldOrigin.DISPATCHER,
        )
    }
    bad_remarks = build_remarks(
        card_fields=bad_fields,
        events=(),
        routing=RoutingDecision(
            routing_rules=VersionRef(name="r", version="1"), rule_id=None, rule_status=None
        ),
        attempt_id=uuid4(),
    )
    bad = grammar_criterion(bad_fields, bad_remarks)
    assert bad.status is CriterionStatus.FAILED and bad.value == 0
    assert bad_fields["description"].raw == "Дым дым из окна квартиры дым дым."  # инвариант 10


# --- замечания D-037: технический сбой и неопределённость источника — не ошибка ученика -----


def test_technical_fault_remark_does_not_count_as_error():
    events = (
        ModelFailure(
            event_id=uuid4(),
            attempt_id=uuid4(),
            seq=1,
            server_ts=T0,
            source="ai_worker",
            component=ComponentName.STT,
            kind=FailureKind.TIMEOUT,
        ),
    )
    remarks = build_remarks(
        card_fields={},
        events=events,
        routing=RoutingDecision(
            routing_rules=VersionRef(name="r", version="1"), rule_id=None, rule_status=None
        ),
        attempt_id=uuid4(),
    )
    fault = next(r for r in remarks if r.type is RemarkType.TECHNICAL_FAULT)
    assert not fault.counts_as_error


def test_source_uncertainty_remark_not_a_student_error():
    from dds_ai.contracts.criteria import RuleStatus

    routing = RoutingDecision(
        routing_rules=VersionRef(name="r", version="1"),
        rule_id="syn-r3",
        rule_status=RuleStatus.QUARANTINED,
    )
    remarks = build_remarks(card_fields={}, events=(), routing=routing, attempt_id=uuid4())
    unc = next(r for r in remarks if r.type is RemarkType.SOURCE_UNCERTAINTY)
    assert not unc.counts_as_error
    assert unc.owner == "expert"


# --- семантический оцениватель: эквивалентность, STT-устойчивость, невалидный выход --------


def _block(prompt: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.S)
    assert m, f"prompt has no <{tag}> block:\n{prompt}"
    return m.group(1)


class ScriptedLLM:
    """Фейковая LLM: возвращает JSON, собранный из содержимого промпта, а не фиксированный
    текст — чтобы цитата всегда оставалась дословной подстрокой проверяемого текста, как и
    требует `LLMSemanticJudge` (иначе результат — `InvalidOutput`)."""

    model_ref = ModelRef(component="llm", model_name="scripted-fake", model_version="0")

    def __init__(self, *, match: str = "full", reason_code: str = "paraphrase_ok"):
        self.match, self.reason_code = match, reason_code
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        self.prompts.append(prompt)
        if "<текст_диспетчера>" in prompt:
            text = _block(prompt, "текст_диспетчера")
            quote = text.split(".")[0][:40].strip()
            return json.dumps(
                {"match": self.match, "quote": quote, "reason_code": self.reason_code},
                ensure_ascii=False,
            )
        transcript = _block(prompt, "транскрипт_диспетчера")
        items_block = _block(prompt, "элементы_доклада")
        item_ids = [
            line.split(":", 1)[0].strip() for line in items_block.splitlines() if line.strip()
        ]
        quote = transcript.split(".")[0][:40].strip()
        return json.dumps(
            {"items": [{"item_id": i, "match": self.match, "quote": quote} for i in item_ids]},
            ensure_ascii=False,
        )


class BrokenLLM:
    model_ref = ModelRef(component="llm", model_name="broken-fake", model_version="0")

    def __init__(self, answer: str):
        self.answer = answer

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        return self.answer


def _field_context(student_text: str) -> dict:
    return {
        "kind": "field_fact",
        "field_name": "description",
        "student_text": student_text,
        "expected_facts": [{"label": "Обстоятельства", "values": ["дым из окна квартиры"]}],
    }


def test_two_paraphrases_of_the_same_fact_get_the_same_score():
    judge = LLMSemanticJudge(ScriptedLLM(match="full"))
    a = judge.judge("card.circumstances", _field_context("Из окна квартиры соседей идёт дым."))
    b = judge.judge(
        "card.circumstances", _field_context("Наблюдается задымление из окна соседней квартиры.")
    )
    assert (a.status, a.value) == (b.status, b.value) == (CriterionStatus.PASSED, 1)


def test_field_fact_confident_none_is_not_checked_by_default_policy():
    """F-2: пока согласие ниже W-03, модель одна не выносит `failed` для `field_fact`, если
    критерий не перечислен в `JudgePolicy.failable_criteria` (пусто по умолчанию) — уверенное
    несовпадение уходит в `not_checked`, ждёт эксперта, не штраф (инвариант 4)."""
    judge = LLMSemanticJudge(ScriptedLLM(match="none", reason_code="wrong_value"))
    assert judge.policy.failable_criteria == frozenset()
    r = judge.judge("card.circumstances", _field_context("Всё в порядке, пожара нет."))
    assert r.status is CriterionStatus.NOT_CHECKED and r.value is None
    assert "wrong_value" in r.explanation
    assert "эксперт" in r.explanation.lower()


def test_field_fact_none_match_fails_when_criterion_is_in_policy():
    """Критерий, явно разрешённый конфигом (`JudgePolicy.failable_criteria`), может получить
    `failed` от модели — конкретное сравнение с указанием причины."""
    policy = JudgePolicy(failable_criteria=frozenset({"card.circumstances"}))
    judge = LLMSemanticJudge(ScriptedLLM(match="none", reason_code="wrong_value"), policy=policy)
    r = judge.judge("card.circumstances", _field_context("Всё в порядке, пожара нет."))
    assert r.status is CriterionStatus.FAILED and r.value == 0
    assert "wrong_value" in r.explanation


def test_field_fact_none_match_not_failed_for_criterion_outside_policy():
    """Конфиг разрешает `failed` только перечисленным критериям — `manual.additions` здесь не
    входит в политику, значит остаётся `not_checked`, даже когда `card.circumstances` разрешён."""
    policy = JudgePolicy(failable_criteria=frozenset({"card.circumstances"}))
    judge = LLMSemanticJudge(ScriptedLLM(match="none", reason_code="wrong_value"), policy=policy)
    r = judge.judge("manual.additions", _field_context("Всё в порядке, пожара нет."))
    assert r.status is CriterionStatus.NOT_CHECKED and r.value is None


def test_unavailable_field_is_not_checked_not_penalized():
    judge = LLMSemanticJudge(ScriptedLLM())
    r = judge.judge("card.circumstances", _field_context(None))
    assert r.status is CriterionStatus.NOT_CHECKED
    assert r.value is None


def test_corrupted_transcript_is_not_checked_not_a_penalty():
    """STT-устойчивость (раздел 8): неразборчивый транскрипт -> not_checked, не failed."""
    judge = LLMSemanticJudge(ScriptedLLM(match="unclear"))
    context = {
        "kind": "conversation_coverage",
        "transcript": [{"event_id": "e1", "text": "###неразборчиво### дым ### квартир ###"}],
        "items": [{"item_id": "circumstances.smoke", "description": "дым из окна"}],
    }
    r = judge.judge("voice.facts_transferred", context)
    assert r.status is CriterionStatus.NOT_CHECKED
    assert r.value is None


def test_conversation_coverage_confirms_paraphrase():
    judge = LLMSemanticJudge(ScriptedLLM(match="full"))
    context = {
        "kind": "conversation_coverage",
        "transcript": [{"event_id": "e1", "text": "Наблюдается задымление из окна квартиры."}],
        "items": [{"item_id": "circumstances.smoke", "description": "дым из окна"}],
    }
    r = judge.judge("voice.facts_transferred", context)
    assert r.status is CriterionStatus.PASSED and r.value == 1
    assert r.evidence and r.evidence[0].ref == "e1"


@pytest.mark.parametrize(
    "answer",
    [
        "не json совсем",
        json.dumps({"match": "full", "quote": "", "reason_code": "not_a_real_code"}),
        json.dumps(
            {
                "match": "full",
                "quote": "выдуманная цитата которой нет",
                "reason_code": "exact_match",
            }
        ),
        json.dumps({"match": "maybe"}),
    ],
)
def test_invalid_judge_output_raises_invalid_output_not_a_score(answer):
    judge = LLMSemanticJudge(BrokenLLM(answer))
    with pytest.raises(InvalidOutput):
        judge.judge("card.circumstances", _field_context("Дым из окна квартиры."))


def test_invalid_judge_output_through_evaluation_is_not_checked_not_zero():
    """Тот же невалидный выход, пройдя через `evaluation.judge_criterion`, даёт not_checked,
    а не 0 (инвариант 4) — сквозная проверка защиты, не только модуля judging."""
    from dds_ai.evaluation import judge_criterion

    judge = LLMSemanticJudge(BrokenLLM("{не json"))
    result = judge_criterion(
        judge, FaultInjector(), "card.circumstances", _field_context("Дым из окна квартиры.")
    )
    assert result.status is CriterionStatus.NOT_CHECKED
    assert result.value is None


def test_conversation_coverage_rejects_mismatched_item_ids():
    class WrongIdsLLM:
        model_ref = ModelRef(component="llm", model_name="x", model_version="0")

        def complete(self, prompt: str, *, max_tokens: int) -> str:
            item = {"item_id": "not_the_requested_id", "match": "full", "quote": ""}
            return json.dumps({"items": [item]})

    judge = LLMSemanticJudge(WrongIdsLLM())
    context = {
        "kind": "conversation_coverage",
        "transcript": [{"event_id": "e1", "text": "дым из окна"}],
        "items": [{"item_id": "circumstances.smoke", "description": "дым из окна"}],
    }
    with pytest.raises(InvalidOutput):
        judge.judge("voice.facts_transferred", context)


# --- C-08: экспертная поправка создаёт новую ScoreVersion с причиной -----------------------


def test_expert_correction_creates_new_score_version_with_reason(fire, rubric):
    s = TrainingSession(fire, attempt_id=uuid4())
    s.notify()
    s.clock.advance(10)
    s.open_card()
    s.edit_card(services=["fire_service", "ambulance"])
    s.decide(DispatcherDecision.RESPOND)
    s.clock.advance(60)
    s.submit(incomplete=True)
    ev = s.evaluate(rubric)

    v1 = ScoreVersion(
        score_version_id=uuid4(),
        attempt_id=s.attempt_id,
        author="auto",
        created_at=T0,
        criterion_results=ev.results,
        summary=ev.summary,
    )
    with pytest.raises(ValidationError):
        ScoreVersion(  # поправка без причины запрещена (C-08/C-09)
            score_version_id=uuid4(),
            attempt_id=s.attempt_id,
            parent_score_version_id=v1.score_version_id,
            author="teacher",
            created_at=T0,
            criterion_results=ev.results,
            summary=ev.summary,
        )
    corrected = tuple(
        r if r.criterion_id != "card.address" else r.model_copy(update={"decided_by": "expert"})
        for r in ev.results
    )
    v2 = ScoreVersion(
        score_version_id=uuid4(),
        attempt_id=s.attempt_id,
        parent_score_version_id=v1.score_version_id,
        author="teacher",
        reason="Эксперт пересмотрел адрес по записи разговора.",
        created_at=T0,
        criterion_results=corrected,
        summary=ev.summary,
    )
    assert v2.parent_score_version_id == v1.score_version_id
    assert v2.reason
    assert v1.score_version_id != v2.score_version_id  # исходная версия не перезаписана (C-09)
