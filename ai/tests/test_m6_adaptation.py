"""Веха M6 (6.7–6.9): подсказки и режимы, сложность, профиль, рекомендации, аудит C-08,
аналитика типичных ошибок группы.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dds_ai.analytics import AttemptErrorRecord, build_group_error_report
from dds_ai.contracts.common import Evidence, EvidenceKind, ModelRef, VersionRef
from dds_ai.contracts.criteria import CriterionStatus
from dds_ai.contracts.difficulty import DifficultyFactor, DifficultyVector, DifficultyWeights
from dds_ai.contracts.events import (
    ComponentName,
    DispatcherDecision,
    FailureKind,
    HelpRequested,
    HintShown,
)
from dds_ai.contracts.mode import AttemptTrack, TrainingMode
from dds_ai.contracts.profile import (
    GroupLevel,
    PoolTask,
    ProfileStatus,
    RecommendationStatus,
    SkillProfile,
    TaskPool,
    TeacherAction,
)
from dds_ai.contracts.remarks import Remark, RemarkType, Severity
from dds_ai.contracts.scoring import ScoreVersion, Verdict
from dds_ai.cycle import Channel, TrainingSession, load_synthetic
from dds_ai.difficulty import band_for, load_weights, propose_vector
from dds_ai.difficulty import score as difficulty_score
from dds_ai.faults import FaultInjector
from dds_ai.hints import HintDenied, explain_errors
from dds_ai.mocks.media import MockMediaTransport
from dds_ai.mocks.scoring import score as rubric_score
from dds_ai.profile import (
    AttemptRecord,
    apply_teacher_action,
    build_profile,
    mark_superseded,
    recommend,
)

from .conftest import all_results, result
from .fakes import FakeTTS, StubJudge

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
RULES = VersionRef(name="m6-rules", version="1")

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


def sv(rubric, attempt_id, *, status=CriterionStatus.PASSED, **overrides) -> ScoreVersion:
    results = tuple(all_results(rubric, status, **overrides))
    return ScoreVersion(
        score_version_id=uuid4(),
        attempt_id=attempt_id,
        author="system",
        created_at=T0,
        criterion_results=results,
        summary=rubric_score(rubric, results),
    )


def pool(*, dds_profile: str = "dds-center") -> TaskPool:
    return TaskPool(
        pool_snapshot_id=uuid4(),
        dds_profile=dds_profile,
        assigned_by="teacher-1",
        published_at=T0,
        tasks=(
            PoolTask(
                task_id="t0",
                dds_profile=dds_profile,
                difficulty_band="low",
                difficulty_rank=0,
                weak_groups=("card",),
            ),
            PoolTask(
                task_id="t1",
                dds_profile=dds_profile,
                difficulty_band="medium",
                difficulty_rank=1,
                weak_groups=("voice",),
            ),
            PoolTask(
                task_id="t2", dds_profile=dds_profile, difficulty_band="high", difficulty_rank=2
            ),
        ),
    )


def profile_ok(trainee, *, dds_profile="dds-center", groups) -> SkillProfile:
    return SkillProfile(
        profile_version_id=uuid4(),
        trainee_id=trainee,
        dds_profile=dds_profile,
        status=ProfileStatus.OK,
        groups=groups,
        source_score_versions=(uuid4(),),
        aggregation_rules=RULES,
        created_at=T0,
    )


class FakeRecommender:
    model_ref = ModelRef(component="recommender", model_name="fake-rec", model_version="0")

    def __init__(self, fail: bool = False):
        self.fail = fail

    def rank(self, pool_task_ids, features):
        if self.fail:
            raise RuntimeError("model crashed")
        return list(pool_task_ids)


# --- 6.7 подсказки и режимы --------------------------------------------------------------


def test_guided_hints_and_pauses_are_events_in_history(fire, rubric):
    s = TrainingSession(fire, mode=TrainingMode.GUIDED, judge=StubJudge())
    s.notify()
    s.show_hint("Откройте карточку в течение 30 секунд.")
    s.clock.advance(5)
    s.open_card()
    s.show_hint("Укажите службы и примите решение.")
    s.edit_card(services=["fire_service", "ambulance"], description="Дым из окна квартиры.")
    s.decide(DispatcherDecision.RESPOND)
    s.dial()
    s.say(FIRE_REPORT)
    s.say("Да, верно.")
    s.clock.advance(150)
    s.submit()
    hints = [e for e in s.events if isinstance(e, HintShown)]
    assert [h.text for h in hints] == [
        "Откройте карточку в течение 30 секунд.",
        "Укажите службы и примите решение.",
    ]


def test_independent_mode_denies_guided_hint(fire):
    s = TrainingSession(fire, mode=TrainingMode.INDEPENDENT)
    with pytest.raises(HintDenied):
        s.show_hint("это подсказка о правильном действии")


def test_help_request_moves_attempt_to_supported_and_is_an_event(fire, rubric):
    s = TrainingSession(fire, mode=TrainingMode.INDEPENDENT, judge=StubJudge())
    assert s.track is AttemptTrack.INDEPENDENT
    s.request_help()
    assert any(isinstance(e, HelpRequested) for e in s.events)
    assert s.track is AttemptTrack.SUPPORTED
    run_fire(s)
    evaluation = s.evaluate(rubric)
    assert evaluation.track is AttemptTrack.SUPPORTED
    assert not evaluation.track.counts_toward_independent_rating


def test_help_request_opens_hints_immediately(fire):
    """G-1: кнопка запроса помощи иначе бесполезна — подсказка доступна сразу после запроса,
    без завершения попытки, но недоступна до запроса."""
    s = TrainingSession(fire, mode=TrainingMode.INDEPENDENT)
    with pytest.raises(HintDenied):
        s.show_hint("подсказка до запроса помощи")
    s.request_help()
    s.show_hint("подсказка после запроса помощи")
    hints = [e for e in s.events if isinstance(e, HintShown)]
    assert [h.text for h in hints] == ["подсказка после запроса помощи"]


def test_request_help_only_applies_to_independent_mode(fire):
    s = TrainingSession(fire, mode=TrainingMode.GUIDED)
    with pytest.raises(RuntimeError):
        s.request_help()


def test_one_scenario_passes_both_modes_with_separate_tracks(fire, rubric):
    """Один сценарий проходит оба режима; результат с поддержкой/guided не смешивается с
    самостоятельным рейтингом (D-041)."""
    guided = TrainingSession(
        fire,
        mode=TrainingMode.GUIDED,
        channel=Channel.VOICE,
        tts=FakeTTS(),
        media=MockMediaTransport(),
        judge=StubJudge(),
    )
    run_fire(guided)
    g = guided.evaluate(rubric)

    independent = TrainingSession(
        fire,
        mode=TrainingMode.INDEPENDENT,
        channel=Channel.VOICE,
        tts=FakeTTS(),
        media=MockMediaTransport(),
        judge=StubJudge(),
    )
    run_fire(independent)
    i = independent.evaluate(rubric)

    assert g.summary.verdict is Verdict.PASSED
    assert i.summary.verdict is Verdict.PASSED
    assert g.track is AttemptTrack.GUIDED
    assert i.track is AttemptTrack.INDEPENDENT
    assert not g.track.counts_toward_independent_rating
    assert i.track.counts_toward_independent_rating


def test_explain_errors_covers_only_student_errors_with_evidence_and_exercise():
    student_error = Remark(
        remark_id="r1",
        attempt_id="a1",
        type=RemarkType.GRAMMAR,
        severity=Severity.ERROR,
        criterion_id="manual.grammar",
        text="Повтор слова в описании.",
        evidence=(Evidence(kind=EvidenceKind.CARD_FIELD, ref="description"),),
        owner="rule",
    )
    technical = Remark(
        remark_id="r2",
        attempt_id="a1",
        type=RemarkType.TECHNICAL_FAULT,
        severity=Severity.ERROR,
        text="stt: timeout",
        evidence=(Evidence(kind=EvidenceKind.EVENT, ref="e1"),),
        owner="rule",
    )
    source_unc = Remark(
        remark_id="r3",
        attempt_id="a1",
        type=RemarkType.SOURCE_UNCERTAINTY,
        severity=Severity.INFO,
        criterion_id="routing.services",
        text="Правило не активно.",
        evidence=(Evidence(kind=EvidenceKind.RULE, ref="r"),),
        owner="expert",
    )
    out = explain_errors((student_error, technical, source_unc))
    assert len(out) == 1
    assert out[0].remark_id == "r1"
    assert out[0].error == student_error.text
    assert out[0].evidence == student_error.evidence
    assert out[0].exercise  # непустое упражнение


# --- 6.8 сложность ------------------------------------------------------------------------


def _equal_weights() -> DifficultyWeights:
    return DifficultyWeights(weights_version="w1", weights={f: 1.0 for f in DifficultyFactor})


def test_same_vector_and_weights_version_give_same_score():
    vector = DifficultyVector(values={f: 1 for f in DifficultyFactor})
    w = _equal_weights()
    s1 = difficulty_score(vector, w, scenario=RULES)
    s2 = difficulty_score(vector, w, scenario=RULES)
    assert s1 == s2
    assert s1.total == pytest.approx(5.0)


def test_per_factor_contribution_is_visible():
    w = _equal_weights()
    vector = DifficultyVector(
        values={
            DifficultyFactor.AMBIGUITY: 2,
            DifficultyFactor.CIRCUMSTANCES: 0,
            DifficultyFactor.SERVICES: 1,
            DifficultyFactor.ADDITIONS: 0,
            DifficultyFactor.SPEECH: 1,
        }
    )
    out = difficulty_score(vector, w, scenario=RULES)
    contrib = {c.factor: c.contribution for c in out.contributions}
    assert contrib[DifficultyFactor.AMBIGUITY] == 2
    assert contrib[DifficultyFactor.CIRCUMSTANCES] == 0
    assert out.total == 4


def test_adding_a_complication_never_decreases_the_sum():
    w = _equal_weights()
    base_values = {f: 0 for f in DifficultyFactor}
    base_total = difficulty_score(DifficultyVector(values=base_values), w, scenario=RULES).total
    for factor in DifficultyFactor:
        for level in (1, 2):
            bumped = dict(base_values)
            bumped[factor] = level
            total = difficulty_score(DifficultyVector(values=bumped), w, scenario=RULES).total
            assert total >= base_total


def test_negative_weight_is_rejected():
    weights = {f: 1.0 for f in DifficultyFactor}
    weights[DifficultyFactor.SPEECH] = -1.0
    with pytest.raises(ValueError):
        DifficultyWeights(weights_version="w1", weights=weights)


def test_propose_vector_covers_all_factors_with_rationale(fire):
    proposal = propose_vector(fire)
    assert {f.factor for f in proposal.factors} == set(DifficultyFactor)
    assert all(f.rationale for f in proposal.factors)
    assert proposal.model_ref is None  # детерминированное правило, не модель


def test_default_weights_and_bands_are_consistent():
    w = load_weights()
    assert set(w.weights) == set(DifficultyFactor)
    assert band_for(0.0) == "low"
    assert band_for(3.9) == "low"
    assert band_for(4.0) == "medium"
    assert band_for(7.0) == "high"


def test_difficulty_weights_version_is_recorded_at_attempt_start(fire, rubric):
    """G-4: версия весов сложности попадает в AttemptVersionSnapshot.difficulty_config;
    зафиксирована при старте попытки (конструктор), а не подставляется задним числом."""
    custom_weights = DifficultyWeights(
        weights_version="difficulty-test-custom-1",
        weights=dict.fromkeys(DifficultyFactor, 1.0),
    )
    s = TrainingSession(fire, difficulty_weights=custom_weights)
    # G-4: значение уже зафиксировано в конструкторе, до сборки полного снимка.
    assert s.difficulty_config.version == "difficulty-test-custom-1"

    snapshot = s.version_snapshot(rubric)
    assert snapshot.attempt_id == s.attempt_id
    assert snapshot.difficulty_config.version == "difficulty-test-custom-1"


def test_difficulty_weights_version_defaults_to_published_config(fire, rubric):
    s = TrainingSession(fire)
    assert s.difficulty_config.version == load_weights().weights_version
    assert s.version_snapshot(rubric).difficulty_config.version == load_weights().weights_version


# --- 6.9 профиль подготовки ----------------------------------------------------------------


def test_new_student_gets_insufficient_data_not_an_invented_level():
    profile = build_profile(
        uuid4(),
        [],
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
    )
    assert profile.status is ProfileStatus.INSUFFICIENT_DATA
    assert profile.groups == ()
    assert profile.source_score_versions == ()


def test_provisional_attempt_does_not_worsen_the_profile(rubric):
    trainee, attempt_id = uuid4(), uuid4()
    provisional = sv(
        rubric,
        attempt_id,
        **{
            "voice.facts_transferred": result(
                "voice.facts_transferred", CriterionStatus.NOT_CHECKED
            )
        },
    )
    assert provisional.summary.verdict is Verdict.PROVISIONAL
    record = AttemptRecord(
        attempt_id, "dds-center", "low", AttemptTrack.INDEPENDENT, provisional, rubric
    )
    profile = build_profile(
        trainee,
        [record],
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
    )
    # единственная попытка была provisional → сегмент пуст, не «недостаточно данных из-за
    # плохого результата», а буквально нет ни одного valid_score наблюдения
    assert profile.status is ProfileStatus.INSUFFICIENT_DATA


def test_valid_attempts_build_group_levels_with_attempts_used(rubric):
    trainee = uuid4()
    records = [
        AttemptRecord(
            uuid4(), "dds-center", "low", AttemptTrack.INDEPENDENT, sv(rubric, uuid4()), rubric
        )
        for _ in range(3)
    ]
    profile = build_profile(
        trainee,
        records,
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
    )
    assert profile.status is ProfileStatus.OK
    assert {g.group_id for g in profile.groups} == {g.group_id for g in rubric.groups}
    assert all(g.level == 1.0 for g in profile.groups)
    assert all(g.attempts_used == 3 for g in profile.groups)
    assert len(profile.source_score_versions) == 3


def test_segment_level_hidden_below_min_attempts_threshold(rubric):
    """G-3: уровень в отчёте преподавателю виден от 3 valid_score-попыток в сегменте;
    меньше — insufficient_data, а не уровень по одной-двум попытках. Порог настраиваемый."""
    trainee = uuid4()
    two_records = [
        AttemptRecord(
            uuid4(), "dds-center", "low", AttemptTrack.INDEPENDENT, sv(rubric, uuid4()), rubric
        )
        for _ in range(2)
    ]
    profile_default = build_profile(
        trainee,
        two_records,
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
    )
    assert profile_default.status is ProfileStatus.INSUFFICIENT_DATA
    assert profile_default.groups == ()

    profile_custom_threshold = build_profile(
        trainee,
        two_records,
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
        min_attempts=2,
    )
    assert profile_custom_threshold.status is ProfileStatus.OK

    three_records = [
        *two_records,
        AttemptRecord(
            uuid4(), "dds-center", "low", AttemptTrack.INDEPENDENT, sv(rubric, uuid4()), rubric
        ),
    ]
    profile_at_threshold = build_profile(
        trainee,
        three_records,
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
    )
    assert profile_at_threshold.status is ProfileStatus.OK
    assert all(g.attempts_used == 3 for g in profile_at_threshold.groups)


def test_profile_segments_are_isolated_by_dds_profile_difficulty_and_track(rubric):
    trainee = uuid4()
    center = AttemptRecord(
        uuid4(), "dds-center", "low", AttemptTrack.INDEPENDENT, sv(rubric, uuid4()), rubric
    )
    river = AttemptRecord(
        uuid4(), "dds-river", "low", AttemptTrack.INDEPENDENT, sv(rubric, uuid4()), rubric
    )
    guided = AttemptRecord(
        uuid4(), "dds-center", "low", AttemptTrack.GUIDED, sv(rubric, uuid4()), rubric
    )
    high = AttemptRecord(
        uuid4(), "dds-center", "high", AttemptTrack.INDEPENDENT, sv(rubric, uuid4()), rubric
    )
    profile = build_profile(
        trainee,
        [center, river, guided, high],
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
        min_attempts=1,  # тест изоляции сегментов, не порога G-3
    )
    assert profile.status is ProfileStatus.OK
    assert all(g.attempts_used == 1 for g in profile.groups)


def test_technically_violated_attempt_does_not_worsen_profile(rubric):
    """C-01/инвариант 13: технический сбой → time.* уходит в technical_error → provisional →
    не проходит фильтр valid_score, профиль не строится по такой попытке."""
    trainee, attempt_id = uuid4(), uuid4()
    technical = sv(
        rubric, attempt_id, **{"time.open": result("time.open", CriterionStatus.TECHNICAL_ERROR)}
    )
    assert technical.summary.verdict is Verdict.PROVISIONAL
    record = AttemptRecord(
        attempt_id, "dds-center", "low", AttemptTrack.INDEPENDENT, technical, rubric
    )
    profile = build_profile(
        trainee,
        [record],
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
    )
    assert profile.status is ProfileStatus.INSUFFICIENT_DATA


# --- 6.9 рекомендации (D-045) ----------------------------------------------------------------


def test_new_student_recommendation_is_unavailable_not_an_invented_level():
    trainee = uuid4()
    profile = SkillProfile(
        profile_version_id=uuid4(),
        trainee_id=trainee,
        dds_profile="dds-center",
        status=ProfileStatus.INSUFFICIENT_DATA,
        aggregation_rules=RULES,
        created_at=T0,
    )
    rec = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=pool(),
        current_rank=None,
        selection_rules=RULES,
        now=T0,
    )
    assert rec.status is RecommendationStatus.UNAVAILABLE
    assert rec.task_id is None
    assert "преподавател" in rec.reason.lower()


def test_recommendation_never_leaves_pool_or_dds_profile(rubric):
    trainee = uuid4()
    groups = (
        GroupLevel(group_id="card", level=0.4, attempts_used=5),
        GroupLevel(group_id="voice", level=0.9, attempts_used=5),
    )
    profile = profile_ok(trainee, groups=groups)
    rec = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=pool(),
        current_rank=0,
        selection_rules=RULES,
        now=T0,
    )
    assert rec.status is RecommendationStatus.AVAILABLE
    assert rec.task_id in rec.pool_task_ids
    assert rec.task_id == "t0"  # слабая группа card, ранг 0 — цель попадания

    mismatched = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=pool(dds_profile="dds-river"),
        current_rank=0,
        selection_rules=RULES,
        now=T0,
    )
    assert mismatched.status is RecommendationStatus.UNAVAILABLE
    assert mismatched.task_id is None


def test_empty_pool_gives_a_clear_reason_not_a_generated_task():
    trainee = uuid4()
    profile = profile_ok(trainee, groups=(GroupLevel(group_id="card", level=1.0, attempts_used=1),))
    empty = TaskPool(
        pool_snapshot_id=uuid4(),
        dds_profile="dds-center",
        assigned_by="t",
        published_at=T0,
        tasks=(),
    )
    rec = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=empty,
        current_rank=0,
        selection_rules=RULES,
        now=T0,
    )
    assert rec.status is RecommendationStatus.UNAVAILABLE
    assert rec.reason


def test_insufficient_evidence_keeps_difficulty_unchanged():
    trainee = uuid4()
    profile = profile_ok(trainee, groups=(GroupLevel(group_id="card", level=1.0, attempts_used=1),))
    rec = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=pool(),
        current_rank=0,
        selection_rules=RULES,
        now=T0,
    )
    chosen = next(t for t in pool().tasks if t.task_id == rec.task_id)
    assert chosen.difficulty_rank == 0  # мало доказательств → сложность не растёт


def test_level_up_limited_to_one_neighbor_with_enough_evidence():
    trainee = uuid4()
    profile = profile_ok(
        trainee, groups=(GroupLevel(group_id="voice", level=1.0, attempts_used=5),)
    )
    rec = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=pool(),
        current_rank=0,
        selection_rules=RULES,
        now=T0,
    )
    chosen = next(t for t in pool().tasks if t.task_id == rec.task_id)
    assert chosen.difficulty_rank <= 1  # не больше, чем на один соседний уровень


def test_recommender_failure_falls_back_to_teacher_sequence():
    trainee = uuid4()
    profile = profile_ok(trainee, groups=(GroupLevel(group_id="card", level=0.5, attempts_used=5),))
    injector = FaultInjector()
    injector.inject(ComponentName.RECOMMENDER, FailureKind.TIMEOUT)
    rec = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=pool(),
        current_rank=0,
        selection_rules=RULES,
        now=T0,
        recommender=FakeRecommender(fail=True),
        injector=injector,
        teacher_sequence=["t1"],
    )
    assert rec.status is RecommendationStatus.AVAILABLE
    assert rec.task_id == "t1"
    assert "преподавател" in rec.reason.lower()


def test_recommender_failure_without_teacher_sequence_is_unavailable_profile_unaffected():
    trainee = uuid4()
    profile = profile_ok(trainee, groups=(GroupLevel(group_id="card", level=0.5, attempts_used=5),))
    injector = FaultInjector()
    injector.inject(ComponentName.RECOMMENDER, FailureKind.ERROR)
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
    assert rec.status is RecommendationStatus.UNAVAILABLE
    assert rec.task_id is None
    # профиль, переданный на вход, не мутирует и не помечается хуже:
    assert profile.status is ProfileStatus.OK
    assert profile.groups[0].level == 0.5


def test_recommender_output_outside_candidates_is_rejected_not_silently_accepted():
    trainee = uuid4()
    profile = profile_ok(trainee, groups=(GroupLevel(group_id="card", level=0.5, attempts_used=5),))

    class RogueRecommender:
        model_ref = ModelRef(component="recommender", model_name="rogue", model_version="0")

        def rank(self, pool_task_ids, features):
            return ["not-in-pool"]

    rec = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=pool(),
        current_rank=0,
        selection_rules=RULES,
        now=T0,
        recommender=RogueRecommender(),
    )
    # невалидный выход рекомендателя — деградация (C-04), не тихая подмена задания
    assert rec.status is RecommendationStatus.UNAVAILABLE


# --- C-08 аудит адаптации -------------------------------------------------------------------


def test_selection_is_reproducible_from_saved_inputs():
    trainee = uuid4()
    profile = profile_ok(trainee, groups=(GroupLevel(group_id="card", level=0.4, attempts_used=5),))
    p = pool()  # тот же снимок пула для обоих прогонов — переменные входы, не новый uuid4()
    rid = uuid4()
    r1 = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=p,
        current_rank=0,
        selection_rules=RULES,
        now=T0,
        recommendation_id=rid,
    )
    r2 = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=p,
        current_rank=0,
        selection_rules=RULES,
        now=T0,
        recommendation_id=rid,
    )
    assert r1 == r2


def test_expert_correction_after_auto_selection_updates_next_profile_only(rubric):
    """C-08: исправить оценку после автовыбора → история видна (r1 не переписан), следующий
    профиль обновляется (v2 отличается от v1), уже выданная рекомендация — исторический факт."""
    trainee = uuid4()
    a, b, c = uuid4(), uuid4(), uuid4()
    records_v1 = [
        AttemptRecord(a, "dds-center", "low", AttemptTrack.INDEPENDENT, sv(rubric, a), rubric),
        AttemptRecord(b, "dds-center", "low", AttemptTrack.INDEPENDENT, sv(rubric, b), rubric),
        AttemptRecord(c, "dds-center", "low", AttemptTrack.INDEPENDENT, sv(rubric, c), rubric),
    ]
    profile_v1 = build_profile(
        trainee,
        records_v1,
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
    )
    p = pool()
    rec1 = recommend(
        trainee_id=trainee,
        profile=profile_v1,
        pool=p,
        current_rank=0,
        selection_rules=RULES,
        now=T0,
    )
    assert rec1.status is RecommendationStatus.AVAILABLE

    # эксперт правит оценку попытки c: адрес оказался неверным
    corrected_c = sv(rubric, c, **{"card.address": result("card.address", CriterionStatus.FAILED)})
    records_v2 = [
        records_v1[0],
        records_v1[1],
        AttemptRecord(c, "dds-center", "low", AttemptTrack.INDEPENDENT, corrected_c, rubric),
    ]
    profile_v2 = build_profile(
        trainee,
        records_v2,
        dds_profile="dds-center",
        difficulty_band="low",
        track=AttemptTrack.INDEPENDENT,
        aggregation_rules=RULES,
        now=T0,
    )
    assert profile_v2.profile_version_id != profile_v1.profile_version_id
    card_v1 = next(g for g in profile_v1.groups if g.group_id == "card").level
    card_v2 = next(g for g in profile_v2.groups if g.group_id == "card").level
    assert card_v2 < card_v1

    # rec1 уже выдана и используется активной попыткой — остаётся историческим фактом
    kept = mark_superseded([rec1], applied_ids={rec1.recommendation_id}, reason="score corrected")
    assert kept[0] == rec1
    assert not kept[0].superseded

    # rec1 ещё не применена — помечается устаревшей, но не переписывается
    superseded = mark_superseded([rec1], applied_ids=set(), reason="score corrected")
    assert superseded[0].superseded
    assert superseded[0].superseded_reason == "score corrected"
    assert superseded[0].task_id == rec1.task_id  # исходное решение видно, не переписано
    assert superseded[0].profile_version_id == rec1.profile_version_id

    rec2 = recommend(
        trainee_id=trainee,
        profile=profile_v2,
        pool=p,
        current_rank=0,
        selection_rules=RULES,
        now=T0,
    )
    assert rec2.profile_version_id == profile_v2.profile_version_id


def test_teacher_action_is_part_of_the_audit_trail():
    trainee = uuid4()
    profile = profile_ok(trainee, groups=(GroupLevel(group_id="card", level=0.5, attempts_used=5),))
    rec = recommend(
        trainee_id=trainee,
        profile=profile,
        pool=pool(),
        current_rank=0,
        selection_rules=RULES,
        now=T0,
    )
    updated = apply_teacher_action(rec, action=TeacherAction.REPLACED, by="teacher-1", at=T0)
    assert updated.teacher_action is TeacherAction.REPLACED
    assert updated.teacher_action_by == "teacher-1"
    assert updated.teacher_action_at == T0
    assert updated.recommendation_id == rec.recommendation_id  # тот же факт, не новый


def test_recommendation_requires_matching_superseded_reason_shape():
    with pytest.raises(ValueError):
        from dds_ai.contracts.profile import Recommendation

        Recommendation(
            recommendation_id=uuid4(),
            trainee_id=uuid4(),
            profile_version_id=None,
            pool_snapshot_id=uuid4(),
            pool_task_ids=("t1",),
            status=RecommendationStatus.UNAVAILABLE,
            reason="r",
            selection_rules=RULES,
            created_at=T0,
            superseded=True,
        )


# --- аналитика типичных ошибок группы ([ТЗ], 6.9) --------------------------------------------


def test_analytics_excludes_technical_faults_and_shows_not_checked_share(rubric):
    attempt_id = uuid4()
    student_error = Remark(
        remark_id="r1",
        attempt_id=str(attempt_id),
        type=RemarkType.FACT_FIELD,
        severity=Severity.ERROR,
        criterion_id="card.address",
        text="неверный адрес",
        evidence=(Evidence(kind=EvidenceKind.CARD_FIELD, ref="address"),),
        owner="rule",
    )
    technical = Remark(
        remark_id="r2",
        attempt_id=str(attempt_id),
        type=RemarkType.TECHNICAL_FAULT,
        severity=Severity.ERROR,
        text="stt: timeout",
        evidence=(Evidence(kind=EvidenceKind.EVENT, ref="e1"),),
        owner="rule",
    )
    results = tuple(
        all_results(
            rubric,
            CriterionStatus.PASSED,
            **{
                "voice.facts_transferred": result(
                    "voice.facts_transferred", CriterionStatus.NOT_CHECKED
                )
            },
        )
    )
    rec = AttemptErrorRecord(
        attempt_id=attempt_id,
        dds_profile="dds-center",
        difficulty_band="low",
        valid_score=True,
        remarks=(student_error, technical),
        criterion_results=results,
    )
    report = build_group_error_report([rec], rubric, now=T0)
    assert report.technical_fault_count == 1
    assert all(b.remark_type is not RemarkType.TECHNICAL_FAULT for b in report.buckets)
    bucket = next(b for b in report.buckets if b.remark_type is RemarkType.FACT_FIELD)
    assert bucket.group_id == "card"
    assert bucket.dds_profile == "dds-center"
    assert bucket.attempt_ids == (attempt_id,)
    assert report.not_checked_count == 1
    assert report.not_checked_ratio == pytest.approx(1 / len(results))


def test_analytics_only_counts_valid_score_attempts_for_error_buckets(rubric):
    attempt_id = uuid4()
    student_error = Remark(
        remark_id="r1",
        attempt_id=str(attempt_id),
        type=RemarkType.SEQUENCE,
        severity=Severity.ERROR,
        criterion_id="process.sequence",
        text="нарушен порядок",
        evidence=(Evidence(kind=EvidenceKind.EVENT, ref="e1"),),
        owner="rule",
    )
    invalid = AttemptErrorRecord(
        attempt_id=attempt_id,
        dds_profile="dds-center",
        difficulty_band="low",
        valid_score=False,
        remarks=(student_error,),
        criterion_results=(),
    )
    report = build_group_error_report([invalid], rubric, now=T0)
    assert report.buckets == ()
    assert report.total_valid_attempts == 0
