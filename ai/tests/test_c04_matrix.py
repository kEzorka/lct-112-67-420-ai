"""Матрица C-04 (6.1, раздел 8 промпта): по отдельности отказ STT, LLM, TTS, оценивателя,
рекомендателя; тайм-аут; переполнение очереди; невалидный JSON от модели.

Для каждой строки матрицы проверяются те же четыре свойства (приёмка 6.1):
- сохранённые события: `ModelFailure` в журнале попытки (STT/TTS/LLM — они меняют ход
  живого звонка) либо запись в журнале сбоев воркера (`FaultInjector.log`/`FailureRecord`
  для оценивателя и рекомендателя — они фоновые и не создают событие звонка);
- продолжение попытки: сбой не бросает исключение наружу, цикл доходит до сдачи;
- статус оценки: `Verdict.PROVISIONAL` (не 0, не выдуманный балл — инвариант 4/5);
- исключение из рейтинга: `summary.valid_score is False` (для рекомендателя — эквивалент
  «профиль не ухудшается», у него нет собственного вердикта попытки).

Тайм-аут и переполнение очереди используются как `FailureKind`, внедрённый через
`FaultInjector` — реальная механика очереди/тайм-аута уже отдельно проверена на уровне
`InferenceWorker` в `test_worker.py`; код вызова (`_emit`, `judge_criterion`, `recommend`) не
различает `kind` сбоя, поэтому для матрицы деградации достаточно подставить нужный `kind`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dds_ai.contracts.common import ModelRef, VersionRef
from dds_ai.contracts.criteria import CriterionResult, CriterionStatus
from dds_ai.contracts.events import ComponentName, DispatcherDecision, FailureKind, ModelFailure
from dds_ai.contracts.profile import (
    GroupLevel,
    PoolTask,
    ProfileStatus,
    RecommendationStatus,
    SkillProfile,
    TaskPool,
)
from dds_ai.contracts.scoring import Verdict
from dds_ai.cycle import Channel, TrainingSession, load_synthetic
from dds_ai.faults import FaultInjector
from dds_ai.mocks.media import MockMediaTransport
from dds_ai.profile import recommend
from dds_ai.worker import InferenceWorker

from .fakes import FakeSTT, FakeTTS

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
RULES = VersionRef(name="c04-matrix-rules", version="1")
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


class FakeLLM:
    model_ref = ModelRef(component="llm", model_name="fake-llm", model_version="0")

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        return '{"action": "acknowledge", "text": "Принято, продолжайте."}'


class StubJudge:
    model_ref = ModelRef(component="semantic_judge", model_name="stub-judge", model_version="0")

    def judge(self, criterion_id: str, context: dict) -> CriterionResult:
        return CriterionResult(criterion_id=criterion_id, status=CriterionStatus.PASSED, value=1)


class BadJudge:
    """Возвращает выход, не проходящий схему `CriterionResult` (невалидный JSON от модели)."""

    model_ref = ModelRef(component="semantic_judge", model_name="bad-judge", model_version="0")

    def judge(self, criterion_id: str, context: dict) -> dict:
        return {"criterion_id": criterion_id, "status": "not-a-real-status"}


class FakeRecommender:
    model_ref = ModelRef(component="recommender", model_name="fake-rec", model_version="0")

    def __init__(self, fail: bool = False):
        self.fail = fail

    def rank(self, pool_task_ids, features):
        if self.fail:
            raise RuntimeError("model crashed")
        return list(pool_task_ids)


def run_fire(s, *, open_after=20, submit_after=170):
    s.notify()
    s.clock.advance(open_after)
    s.open_card()
    s.edit_card(
        services=["fire_service", "ambulance"],
        description="Дым из окна квартиры соседей, пострадавших пока не выявлено.",
    )
    s.decide(DispatcherDecision.RESPOND)
    s.dial()
    s.say(FIRE_REPORT)
    s.say("Да, верно.")
    s.clock.advance(submit_after)
    s.submit()


def pool() -> TaskPool:
    return TaskPool(
        pool_snapshot_id=uuid4(),
        dds_profile="dds-center",
        assigned_by="teacher-1",
        published_at=T0,
        tasks=(
            PoolTask(
                task_id="t0", dds_profile="dds-center", difficulty_band="low", difficulty_rank=0
            ),
        ),
    )


def profile_ok(trainee) -> SkillProfile:
    return SkillProfile(
        profile_version_id=uuid4(),
        trainee_id=trainee,
        dds_profile="dds-center",
        status=ProfileStatus.OK,
        groups=(GroupLevel(group_id="card", level=0.5, attempts_used=5),),
        source_score_versions=(uuid4(),),
        aggregation_rules=RULES,
        created_at=T0,
    )


# --- STT/TTS ---------------------------------------------------------------------------------
#
# Preflight — снимок готовности (D-3), не отдельный вызов: `available()` теряет вид сбоя,
# внедрённого через `FaultInjector`, и `preflight_failures()` сводит любую недоступность к
# `error` (`overflow` — только когда исполнитель действительно перегружен очередью, что уже
# проверено на уровне `InferenceWorker` в `test_worker.py`). Поэтому здесь — одна строка на
# компонент; тайм-аут и переполнение очереди с сохранением вида сбоя проверены ниже на LLM,
# оценивателе и рекомендателе — там сбой ловится в момент вызова, а не в preflight-снимке.


def test_stt_failure_matrix(fire, rubric):
    worker = InferenceWorker.from_profile("cpu")
    worker.injector.inject(ComponentName.STT, FailureKind.ERROR)
    s = TrainingSession(
        fire,
        channel=Channel.VOICE,
        stt=FakeSTT(),
        tts=FakeTTS(),
        media=MockMediaTransport(),
        worker=worker,
    )
    # события: сбой STT пишет воркер в preflight (D-3), попытка сразу текстовая (C-04)
    failures = [e for e in s.events if isinstance(e, ModelFailure)]
    assert any(f.component == ComponentName.STT and f.kind is FailureKind.ERROR for f in failures)
    assert s.channel is Channel.TEXT

    run_fire(s)  # продолжение: попытка доходит до сдачи, не падает
    ev = s.evaluate(rubric)
    assert ev.summary.verdict is Verdict.PROVISIONAL  # статус оценки
    assert not ev.summary.valid_score  # исключение из сопоставимого рейтинга


def test_tts_failure_matrix(fire, rubric):
    worker = InferenceWorker.from_profile("cpu")
    worker.injector.inject(ComponentName.TTS, FailureKind.ERROR)
    s = TrainingSession(
        fire, channel=Channel.VOICE, tts=FakeTTS(), media=MockMediaTransport(), worker=worker
    )
    failures = [e for e in s.events if isinstance(e, ModelFailure)]
    assert any(f.component == ComponentName.TTS and f.kind is FailureKind.ERROR for f in failures)
    assert s.channel is Channel.TEXT

    run_fire(s)
    ev = s.evaluate(rubric)
    assert ev.summary.verdict is Verdict.PROVISIONAL
    assert not ev.summary.valid_score


# --- LLM ------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [FailureKind.ERROR, FailureKind.TIMEOUT, FailureKind.OVERFLOW])
def test_llm_failure_matrix(fire, rubric, kind):
    s = TrainingSession(fire, llm=FakeLLM(), judge=StubJudge())
    s.worker.injector.inject(ComponentName.LLM, kind)

    run_fire(s)  # LLM недоступна -> детерминированный сценарный fallback (6.1), не исключение
    failures = [e for e in s.events if isinstance(e, ModelFailure)]
    assert any(f.component == ComponentName.LLM and f.kind == kind for f in failures)

    ev = s.evaluate(rubric)
    assert ev.summary.verdict is Verdict.PROVISIONAL
    assert not ev.summary.valid_score


# --- семантический оцениватель ---------------------------------------------------------------


@pytest.mark.parametrize("kind", [FailureKind.ERROR, FailureKind.TIMEOUT, FailureKind.OVERFLOW])
def test_semantic_judge_failure_matrix(fire, rubric, kind):
    s = TrainingSession(fire, judge=StubJudge())
    s.worker.injector.inject(ComponentName.SEMANTIC_JUDGE, kind)

    run_fire(s)  # сбой фоновый: звонок и сдача идут своим чередом, событий звонка не меняет
    ev = s.evaluate(rubric)
    semantic = [
        r for r in ev.results if r.criterion_id in ("card.circumstances", "manual.additions")
    ]
    assert semantic and all(r.status is CriterionStatus.NOT_CHECKED for r in semantic)
    # события: журнал сбоев воркера (оцениватель — фоновая полоса, без ModelFailure в звонке)
    assert any(
        rec.component is ComponentName.SEMANTIC_JUDGE and rec.kind is kind
        for rec in s.worker.injector.log
    )
    assert ev.summary.verdict is Verdict.PROVISIONAL
    assert not ev.summary.valid_score


def test_semantic_judge_invalid_json_matrix(fire, rubric):
    """Невалидный JSON от модели: выход не проходит схему `CriterionResult` -> not_checked."""
    s = TrainingSession(fire, judge=BadJudge())
    run_fire(s)
    ev = s.evaluate(rubric)
    semantic = [
        r for r in ev.results if r.criterion_id in ("card.circumstances", "manual.additions")
    ]
    assert semantic and all(r.status is CriterionStatus.NOT_CHECKED for r in semantic)
    assert any(
        rec.component is ComponentName.SEMANTIC_JUDGE and rec.kind is FailureKind.INVALID_OUTPUT
        for rec in s.worker.injector.log
    )
    assert ev.summary.verdict is Verdict.PROVISIONAL
    assert not ev.summary.valid_score


# --- рекомендатель ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [FailureKind.ERROR, FailureKind.TIMEOUT, FailureKind.OVERFLOW])
def test_recommender_failure_matrix(kind):
    trainee = uuid4()
    profile = profile_ok(trainee)
    injector = FaultInjector()
    injector.inject(ComponentName.RECOMMENDER, kind)

    rec = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=pool(),
        current_rank=0,
        selection_rules=RULES,
        now=T0,
        recommender=FakeRecommender(fail=True),
        injector=injector,
    )
    # события: журнал сбоев воркера
    assert any(r.component is ComponentName.RECOMMENDER and r.kind is kind for r in injector.log)
    # продолжение: recommend() не бросает исключение, возвращает решение
    assert rec.status is RecommendationStatus.UNAVAILABLE
    assert rec.task_id is None
    assert "рекомендатель" in rec.reason.lower() or "рекоменд" in rec.reason.lower()
    # «рейтинг»/профиль не ухудшается: тот же объект возвращён вызывающему без изменений
    assert profile.status is ProfileStatus.OK
    assert profile.groups[0].level == 0.5
