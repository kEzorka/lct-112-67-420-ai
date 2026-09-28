"""Inference-worker: политика адаптеров, журнал сбоев, preflight (6.1, C-04, D-058)."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, Field

from .common import Contract, NonEmptyStr
from .events import ComponentName, FailureKind


class Lane(StrEnum):
    """Синхронная нагрузка живого звонка отделена от фоновой."""

    INTERACTIVE = "interactive"
    BACKGROUND = "background"


class AdapterPolicy(Contract):
    component: ComponentName
    lane: Lane
    timeout_s: float = Field(gt=0)
    max_queue: int = Field(ge=1)
    max_concurrency: int = Field(ge=1)
    failure_policy_version: NonEmptyStr


class FailureRecord(Contract):
    component: ComponentName
    kind: FailureKind
    at: AwareDatetime
    attempt_id: UUID | None = None
    affected_evidence: tuple[NonEmptyStr, ...] = ()
    detail: str | None = None


class ComponentHealth(StrEnum):
    UP = "up"
    OVERLOADED = "overloaded"
    DOWN = "down"


class ComponentStatus(Contract):
    component: ComponentName
    health: ComponentHealth
    model_version: str | None = None
    queue_length: int = Field(default=0, ge=0)
    lane: Lane | None = Field(default=None, description="Полоса исполнителя; None — вне воркера")
    running: int = Field(default=0, ge=0)


class PreflightReport(Contract):
    profile: NonEmptyStr = Field(description="cpu | gpu")
    components: tuple[ComponentStatus, ...]
    voice_path_ok: bool

    @property
    def ready(self) -> bool:
        return self.voice_path_ok and all(c.health is ComponentHealth.UP for c in self.components)
