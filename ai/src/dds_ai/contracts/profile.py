"""Профиль подготовки и рекомендации (D-044, D-045, C-07, C-08)."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from .common import Contract, ModelRef, NonEmptyStr, VersionRef
from .mode import AttemptTrack


class ProfileStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"


class GroupLevel(Contract):
    group_id: NonEmptyStr
    level: float = Field(ge=0, le=1)
    attempts_used: int = Field(ge=1)


class SkillProfile(Contract):
    """Профиль по пяти группам рубрики, отдельно по профилю ДДС, сложности и режиму (D-044).

    `difficulty_band`/`track` — сегмент, из которого построен именно этот профиль; полный
    профиль ученика — набор `SkillProfile` по разным сегментам, не один документ (6.9).
    """

    profile_version_id: UUID
    trainee_id: UUID
    dds_profile: NonEmptyStr
    difficulty_band: NonEmptyStr = Field(
        default="all", description="Сегмент по сложности; «all» — без разбивки по сложности"
    )
    track: AttemptTrack = Field(
        default=AttemptTrack.INDEPENDENT, description="Сегмент по режиму (D-041)"
    )
    status: ProfileStatus
    groups: tuple[GroupLevel, ...] = ()
    source_score_versions: tuple[UUID, ...] = ()
    aggregation_rules: VersionRef
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _status(self) -> SkillProfile:
        if self.status is ProfileStatus.INSUFFICIENT_DATA and self.groups:
            raise ValueError("insufficient_data profile must not claim levels")
        if self.status is ProfileStatus.OK and not (self.groups and self.source_score_versions):
            raise ValueError("profile levels require source score versions")
        return self


class RecommendationStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class TeacherAction(StrEnum):
    """Действие преподавателя над рекомендацией (C-08: аудит хранит и его)."""

    ACCEPTED = "accepted"
    REPLACED = "replaced"
    DISABLED = "disabled"


class PoolTask(Contract):
    """Задание опубликованного пула (D-045): рекомендатель не выбирает вне этого списка."""

    task_id: NonEmptyStr
    dds_profile: NonEmptyStr
    difficulty_band: NonEmptyStr
    difficulty_rank: int = Field(
        ge=0, description="Порядок сложности внутри dds_profile (0 — самый лёгкий)"
    )
    weak_groups: tuple[NonEmptyStr, ...] = Field(
        default=(), description="Группы рубрики, которые задание тренирует"
    )


class TaskPool(Contract):
    """Пул, назначенный преподавателем (D-045); снимок фиксируется в `Recommendation` (C-08)."""

    pool_snapshot_id: UUID
    dds_profile: NonEmptyStr
    assigned_by: NonEmptyStr
    tasks: tuple[PoolTask, ...] = ()
    published_at: AwareDatetime

    @model_validator(mode="after")
    def _shape(self) -> TaskPool:
        ids = [t.task_id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task_id must be unique within a pool")
        foreign = {t.task_id for t in self.tasks if t.dds_profile != self.dds_profile}
        if foreign:
            raise ValueError(f"pool tasks must match pool dds_profile: {sorted(foreign)}")
        return self


class Recommendation(Contract):
    """Рекомендация следующего задания (D-045) с полным аудитом выбора (C-08)."""

    recommendation_id: UUID
    trainee_id: UUID
    profile_version_id: UUID | None
    source_score_versions: tuple[UUID, ...] = Field(
        default=(), description="Версии оценок, на которых построен исходный профиль (C-08)"
    )
    pool_snapshot_id: UUID
    pool_task_ids: tuple[NonEmptyStr, ...]
    status: RecommendationStatus
    task_id: str | None = None
    reason: NonEmptyStr
    model_ref: ModelRef | None = None
    selection_rules: VersionRef
    auto_applied: bool = False
    superseded: bool = Field(
        default=False,
        description="Ещё не применённая рекомендация устарела после пересчёта оценки (C-08)",
    )
    superseded_reason: str | None = None
    teacher_action: TeacherAction | None = None
    teacher_action_by: NonEmptyStr | None = None
    teacher_action_at: AwareDatetime | None = None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _pool(self) -> Recommendation:
        if self.status is RecommendationStatus.AVAILABLE:
            if self.task_id is None:
                raise ValueError("available recommendation requires task_id")
            if self.task_id not in self.pool_task_ids:
                raise ValueError("recommended task must come from the assigned pool")
        elif self.task_id is not None:
            raise ValueError("unavailable recommendation must not name a task")
        if self.superseded != (self.superseded_reason is not None):
            raise ValueError("superseded_reason is required exactly when superseded is true")
        if (self.teacher_action is None) != (self.teacher_action_by is None):
            raise ValueError("teacher_action requires teacher_action_by and vice versa")
        if (self.teacher_action is None) != (self.teacher_action_at is None):
            raise ValueError("teacher_action requires teacher_action_at and vice versa")
        return self
