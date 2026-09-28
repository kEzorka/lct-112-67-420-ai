"""Аналитика типичных ошибок группы для преподавателя (6.9, [ТЗ]).

Считается только по `valid_score` (инвариант 13, C-07); технические сбои и доля
«не проверено» — отдельные, честные числа, не растворённые в счётчике ошибок (вывод
калибровки M5: `docs/ai/calibration-report.md` — на пилоте много `not_checked`).
Каждая цифра раскрывается до попыток (`attempt_ids`).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from .common import Contract, NonEmptyStr
from .remarks import RemarkType


class ErrorFrequencyBucket(Contract):
    remark_type: RemarkType
    group_id: NonEmptyStr | None = Field(
        default=None, description="Группа рубрики критерия замечания; None — без привязки"
    )
    dds_profile: NonEmptyStr
    difficulty_band: NonEmptyStr
    count: int = Field(ge=1, description="Число замечаний этого типа/группы в сегменте")
    attempt_ids: tuple[UUID, ...] = Field(min_length=1, description="Раскрытие до попыток")

    @model_validator(mode="after")
    def _shape(self) -> ErrorFrequencyBucket:
        if len(set(self.attempt_ids)) > self.count:
            raise ValueError("count cannot be smaller than the number of distinct attempts")
        return self


class GroupErrorReport(Contract):
    generated_at: AwareDatetime
    rubric_version: NonEmptyStr
    buckets: tuple[ErrorFrequencyBucket, ...] = ()
    technical_fault_count: int = Field(
        ge=0, description="Технические сбои — отдельно, не ошибка ученика/группы (инвариант 4)"
    )
    not_checked_count: int = Field(ge=0, description="Результаты критериев со статусом not_checked")
    total_criteria_count: int = Field(
        ge=0, description="Все рассмотренные результаты критериев (не только valid_score)"
    )
    total_valid_attempts: int = Field(ge=0, description="Попытки с valid_score, вошедшие в buckets")

    @model_validator(mode="after")
    def _shape(self) -> GroupErrorReport:
        if self.not_checked_count > self.total_criteria_count:
            raise ValueError("not_checked_count cannot exceed total_criteria_count")
        return self

    @property
    def not_checked_ratio(self) -> float | None:
        """Явная доля «не проверено» — не прячется внутри счётчика ошибок (вывод M5)."""
        if self.total_criteria_count == 0:
            return None
        return round(self.not_checked_count / self.total_criteria_count, 6)
