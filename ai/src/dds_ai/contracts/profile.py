"""Профиль подготовки и рекомендации (D-044, D-045, C-07, C-08)."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from .common import Contract, ModelRef, NonEmptyStr, VersionRef


class ProfileStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"


class GroupLevel(Contract):
    group_id: NonEmptyStr
    level: float = Field(ge=0, le=1)
    attempts_used: int = Field(ge=1)


class SkillProfile(Contract):
    profile_version_id: UUID
    trainee_id: UUID
    dds_profile: NonEmptyStr
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


class Recommendation(Contract):
    recommendation_id: UUID
    trainee_id: UUID
    profile_version_id: UUID | None
    pool_snapshot_id: UUID
    pool_task_ids: tuple[NonEmptyStr, ...]
    status: RecommendationStatus
    task_id: str | None = None
    reason: NonEmptyStr
    model_ref: ModelRef | None = None
    selection_rules: VersionRef
    auto_applied: bool = False
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
        return self
