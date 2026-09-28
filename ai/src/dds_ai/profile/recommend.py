"""Выбор следующего задания (D-045, 6.9, C-07, C-08).

Детерминированное правило — основа, всегда доступна. `Recommender` (модель ранжирования) —
опциональная надстройка через `FaultInjector`/`InferenceWorker`: если он включён и
отказывает, рекомендация не понижает профиль, а либо берёт заранее назначенную
преподавателем последовательность, либо явно становится «недоступна» (C-04). Рекомендатель
никогда не предлагает задание вне снимка пула.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID, uuid4

from ..contracts.common import VersionRef
from ..contracts.events import ComponentName
from ..contracts.profile import (
    PoolTask,
    ProfileStatus,
    Recommendation,
    RecommendationStatus,
    SkillProfile,
    TaskPool,
    TeacherAction,
)
from ..faults import ComponentFailure, FaultInjector, InvalidOutput
from ..ports import Recommender

# «Первые занятия» (D-045): пока на сегмент меньше попыток, сложность не поднимается ни на
# один уровень — рабочая конвенция команды, не пункт реестра решений.
MIN_ATTEMPTS_FOR_LEVEL_UP = 3


def _weakest_groups(profile: SkillProfile) -> tuple[str, ...]:
    if not profile.groups:
        return ()
    weakest = min(g.level for g in profile.groups)
    return tuple(sorted(g.group_id for g in profile.groups if g.level == weakest))


def _pick(candidates: Sequence[PoolTask], weak_groups: tuple[str, ...], base_rank: int) -> PoolTask:
    def key(t: PoolTask) -> tuple[int, int, str]:
        targets_weak = 0 if weak_groups and set(t.weak_groups) & set(weak_groups) else 1
        return (targets_weak, abs(t.difficulty_rank - base_rank), t.task_id)

    return min(candidates, key=key)


def _reason(chosen: PoolTask, weak_groups: tuple[str, ...], allow_up: bool, base_rank: int) -> str:
    parts = [f"Задание «{chosen.task_id}» (сложность {chosen.difficulty_rank})"]
    if weak_groups and set(chosen.weak_groups) & set(weak_groups):
        parts.append(f"тренирует слабые группы: {', '.join(weak_groups)}")
    if chosen.difficulty_rank > base_rank:
        parts.append("повышение на один соседний уровень" if allow_up else "уровень сохранён")
    else:
        parts.append("уровень сохранён")
    return "; ".join(parts) + "."


def recommend(
    *,
    trainee_id: UUID,
    profile: SkillProfile,
    pool: TaskPool,
    current_rank: int | None,
    selection_rules: VersionRef,
    now: datetime,
    recommendation_id: UUID | None = None,
    auto_apply_enabled: bool = False,
    teacher_sequence: Sequence[str] | None = None,
    recommender: Recommender | None = None,
    injector: FaultInjector | None = None,
) -> Recommendation:
    """Чисто функциональный выбор: одинаковые входы (включая `recommendation_id`/`now`) дают
    одинаковый результат (C-08: «прогон по сохранённым входам воспроизводит решение»)."""
    common = {
        "recommendation_id": recommendation_id or uuid4(),
        "trainee_id": trainee_id,
        "profile_version_id": profile.profile_version_id,
        "source_score_versions": profile.source_score_versions,
        "pool_snapshot_id": pool.pool_snapshot_id,
        "pool_task_ids": tuple(t.task_id for t in pool.tasks),
        "selection_rules": selection_rules,
        "created_at": now,
    }

    def unavailable(reason: str) -> Recommendation:
        return Recommendation(**common, status=RecommendationStatus.UNAVAILABLE, reason=reason)

    if profile.dds_profile != pool.dds_profile:
        return unavailable(
            f"Пул назначен для профиля ДДС «{pool.dds_profile}», "
            f"у ученика — «{profile.dds_profile}»."
        )
    if not pool.tasks:
        return unavailable("Пул заданий пуст: назначение ждёт преподавателя.")
    if profile.status is ProfileStatus.INSUFFICIENT_DATA:
        return unavailable(
            "Недостаточно данных по этому сегменту: начальный уровень назначает преподаватель."
        )

    weak = _weakest_groups(profile)
    evidence = max((g.attempts_used for g in profile.groups), default=0)
    base_rank = current_rank if current_rank is not None else 0
    allow_up = evidence >= MIN_ATTEMPTS_FOR_LEVEL_UP
    max_rank = base_rank + 1 if allow_up else base_rank
    candidates = [t for t in pool.tasks if t.difficulty_rank <= max_rank]
    if not candidates:
        return unavailable("В назначенном пуле нет задания подходящей сложности.")

    chosen = _pick(candidates, weak, base_rank)
    reason = _reason(chosen, weak, allow_up, base_rank)
    model_ref = None

    if recommender is not None:
        inj = injector or FaultInjector()
        candidate_ids = [t.task_id for t in candidates]

        def run() -> list[str]:
            ranked = recommender.rank(
                candidate_ids, {"weak_groups": weak, "current_rank": base_rank}
            )
            if not ranked or ranked[0] not in candidate_ids:
                raise InvalidOutput("recommender picked a task outside the rule's candidate set")
            return ranked

        try:
            ranked = inj.call(ComponentName.RECOMMENDER, run)
        except ComponentFailure:
            if teacher_sequence:
                fallback = next((tid for tid in teacher_sequence if tid in candidate_ids), None)
                if fallback is None:
                    return unavailable(
                        "Рекомендатель недоступен (C-04); в последовательности преподавателя "
                        "нет подходящего задания."
                    )
                chosen = next(t for t in candidates if t.task_id == fallback)
                reason = (
                    "Рекомендатель недоступен (C-04): использована последовательность, "
                    f"назначенная преподавателем — «{chosen.task_id}»."
                )
            else:
                return unavailable(
                    "Рекомендатель недоступен (C-04), последовательность "
                    "от преподавателя не назначена."
                )
        else:
            chosen = next(t for t in candidates if t.task_id == ranked[0])
            model_ref = recommender.model_ref
            reason = (
                f"Модель ранжирования выбрала «{chosen.task_id}» "
                f"среди кандидатов правила ({reason})"
            )

    return Recommendation(
        **common,
        status=RecommendationStatus.AVAILABLE,
        task_id=chosen.task_id,
        reason=reason,
        model_ref=model_ref,
        auto_applied=auto_apply_enabled,
    )


def mark_superseded(
    recommendations: Sequence[Recommendation], *, applied_ids: set[UUID], reason: str
) -> list[Recommendation]:
    """C-08: применённая рекомендация — исторический факт; неприменённые — устаревают."""
    out = []
    for r in recommendations:
        if r.recommendation_id in applied_ids or r.superseded:
            out.append(r)
            continue
        data = r.model_dump()
        data.update(superseded=True, superseded_reason=reason)
        out.append(Recommendation.model_validate(data))
    return out


def apply_teacher_action(
    recommendation: Recommendation, *, action: TeacherAction, by: str, at: datetime
) -> Recommendation:
    """C-08: действие преподавателя — часть аудита рекомендации."""
    data = recommendation.model_dump()
    data.update(teacher_action=action, teacher_action_by=by, teacher_action_at=at)
    return Recommendation.model_validate(data)
