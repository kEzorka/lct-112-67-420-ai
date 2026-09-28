"""Профиль подготовки по пяти группам рубрики (D-044, G-3, 6.9, C-07, инвариант 13).

Вход — только `valid_score` попытки (эту фильтрацию делает сам `build_profile`: попытки
`provisional`/`not_scored` не передаются вообще или отбрасываются) и экспертные поправки —
поправка приходит как более новая версия `ScoreVersion` для той же попытки, вызывающий код
просто передаёт актуальную версию (`AttemptRecord.score_version`), отдельного кода коррекции
здесь не нужно: append-only история версий уже гарантирует, что «новая версия» и есть
поправка.

Уровень сегмента (для отчёта преподавателю) виден только от `min_attempts` `valid_score`-
попыток в сегменте (по умолчанию 3, настраивается вызывающим кодом — G-3); меньше —
`insufficient_data`, чтобы не показывать уровень по одной-двум попыткам.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from ..contracts.common import VersionRef
from ..contracts.mode import AttemptTrack
from ..contracts.profile import GroupLevel, ProfileStatus, SkillProfile
from ..contracts.rubric import Rubric
from ..contracts.scoring import ScoreVersion

DEFAULT_MIN_SEGMENT_ATTEMPTS = 3


@dataclass(frozen=True)
class AttemptRecord:
    """Одна попытка, сведённая для построения профиля: сегмент + актуальная версия оценки."""

    attempt_id: UUID
    dds_profile: str
    difficulty_band: str
    track: AttemptTrack
    score_version: ScoreVersion
    rubric: Rubric


def _segment(
    attempts: Sequence[AttemptRecord],
    *,
    dds_profile: str,
    difficulty_band: str,
    track: AttemptTrack,
):
    return [
        a
        for a in attempts
        if a.dds_profile == dds_profile
        and a.difficulty_band == difficulty_band
        and a.track is track
        and a.score_version.summary.valid_score  # C-07/инвариант 13: только passed/not_passed
    ]


def build_profile(
    trainee_id: UUID,
    attempts: Sequence[AttemptRecord],
    *,
    dds_profile: str,
    difficulty_band: str,
    track: AttemptTrack,
    aggregation_rules: VersionRef,
    now: datetime,
    profile_version_id: UUID | None = None,
    min_attempts: int = DEFAULT_MIN_SEGMENT_ATTEMPTS,
) -> SkillProfile:
    """Профиль одного сегмента (dds_profile × сложность × режим), только из `valid_score`.

    Технически нарушенные, `provisional` и `not_checked`-насыщенные попытки не проходят
    фильтр `valid_score` и, значит, профиль не ухудшают (инвариант 13) — не потому, что их
    результат отбрасывается избирательно, а потому что такая попытка структурно не может
    получить `passed`/`not_passed` (C-06: непроверенный критерий → `provisional`).

    Уровень сегмента не показывается при менее чем `min_attempts` `valid_score`-попыток в
    сегменте (G-3, по умолчанию 3) — профиль остаётся `insufficient_data`.
    """
    segment = _segment(
        attempts, dds_profile=dds_profile, difficulty_band=difficulty_band, track=track
    )
    pv_id = profile_version_id or uuid4()
    if len(segment) < min_attempts:
        return SkillProfile(
            profile_version_id=pv_id,
            trainee_id=trainee_id,
            dds_profile=dds_profile,
            difficulty_band=difficulty_band,
            track=track,
            status=ProfileStatus.INSUFFICIENT_DATA,
            aggregation_rules=aggregation_rules,
            created_at=now,
        )

    values: dict[str, list[float]] = defaultdict(list)
    attempt_ids: dict[str, set[UUID]] = defaultdict(set)
    for a in segment:
        for cr in a.score_version.criterion_results:
            if cr.value is None:
                continue
            try:
                group_id = a.rubric.group_of(cr.criterion_id).group_id
            except KeyError:
                continue
            values[group_id].append(float(cr.value))
            attempt_ids[group_id].add(a.attempt_id)

    groups = tuple(
        GroupLevel(
            group_id=g,
            level=round(sum(values[g]) / len(values[g]), 6),
            attempts_used=len(attempt_ids[g]),
        )
        for g in sorted(values)
    )
    if not groups:
        return SkillProfile(
            profile_version_id=pv_id,
            trainee_id=trainee_id,
            dds_profile=dds_profile,
            difficulty_band=difficulty_band,
            track=track,
            status=ProfileStatus.INSUFFICIENT_DATA,
            aggregation_rules=aggregation_rules,
            created_at=now,
        )

    source_score_versions = tuple(dict.fromkeys(a.score_version.score_version_id for a in segment))
    return SkillProfile(
        profile_version_id=pv_id,
        trainee_id=trainee_id,
        dds_profile=dds_profile,
        difficulty_band=difficulty_band,
        track=track,
        status=ProfileStatus.OK,
        groups=groups,
        source_score_versions=source_score_versions,
        aggregation_rules=aggregation_rules,
        created_at=now,
    )
