"""Жизненный цикл черновика сценария (6.5): черновик → проверка → утверждён → опубликован →
архив. Комментарий преподавателя порождает исправленный черновик (новую версию, `revises`
ссылается на предыдущую). Публикация не обходит gate C-03 (`scenarios/lifecycle.py`).

Контракт лежит отдельно от `scenario.py`, чтобы M1-контракт сценария (уже используемый
узлами A/D) не обрастал полями жизненного цикла черновика.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from .common import Contract, ModelRef, NonEmptyStr, VersionRef
from .scenario import Scenario


class ScenarioStatus(StrEnum):
    """Жизненный цикл черновика (6.5): draft → validated → approved → published → archived."""

    DRAFT = "draft"
    VALIDATED = "validated"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


ASSIGNABLE_STATUSES = frozenset({ScenarioStatus.PUBLISHED})


class GenerationInputs(Contract):
    """Вход генератора черновика: параметры от преподавателя (6.5)."""

    incident_type: NonEmptyStr
    location: NonEmptyStr
    difficulty: NonEmptyStr
    topic: NonEmptyStr
    dds_profile: NonEmptyStr = "profile-1"


class GenerationMeta(Contract):
    """Основание генерации (инвариант 6): версия модели/шаблона и использованные фрагменты."""

    inputs: GenerationInputs
    model_ref: ModelRef | None = Field(
        default=None, description="None — детерминированный шаблон (базовая реализация 6.5.5)"
    )
    template_version: NonEmptyStr | None = Field(
        default=None, description="Версия шаблона, если генерация детерминированная"
    )
    corpus_version: VersionRef | None = None
    used_fragment_ids: tuple[NonEmptyStr, ...] = Field(
        default=(), description="Фрагменты базы знаний (6.10), использованные при генерации"
    )

    @model_validator(mode="after")
    def _one_generator(self) -> GenerationMeta:
        if self.model_ref is None and self.template_version is None:
            raise ValueError("generation requires either model_ref or template_version")
        return self


class ScenarioDraftRecord(Contract):
    """Черновик/версия сценария с её местом в жизненном цикле (6.5)."""

    scenario: Scenario
    status: ScenarioStatus
    generation: GenerationMeta
    revises: VersionRef | None = Field(
        default=None, description="Предыдущая версия, которую исправляет комментарий (6.5)"
    )
    teacher_comment: str | None = Field(
        default=None, description="Комментарий преподавателя, породивший этот черновик"
    )
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _consistent(self) -> ScenarioDraftRecord:
        if self.status is not ScenarioStatus.DRAFT and self.revises is not None:
            raise ValueError("revises is only set on a freshly revised draft")
        if self.revises is not None and self.teacher_comment is None:
            raise ValueError("a revision must carry the teacher comment that triggered it")
        return self

    @property
    def assignable(self) -> bool:
        return self.status in ASSIGNABLE_STATUSES
