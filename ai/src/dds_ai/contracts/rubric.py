"""Рубрика как версионируемая конфигурация (C-06, W-01).

Названия групп, веса, порог и список критических критериев — данные, а не код:
ответ заказчика на Q05 меняет конфигурацию, а не модуль.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from .common import Contract, NonEmptyStr


class RubricCriterion(Contract):
    criterion_id: NonEmptyStr
    title: NonEmptyStr
    weight: float = Field(gt=0, description="Внутригрупповой вес, нормируется (C-06)")
    critical: bool = Field(default=False, description="Проверенный провал = «не сдано» (W-01)")


class RubricGroup(Contract):
    group_id: NonEmptyStr
    title: NonEmptyStr
    weight: float = Field(gt=0)
    criteria: tuple[RubricCriterion, ...] = Field(min_length=1)


class Rubric(Contract):
    rubric_version: NonEmptyStr
    pass_threshold: float = Field(ge=0, le=100)
    groups: tuple[RubricGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Rubric:
        total = sum(g.weight for g in self.groups)
        if abs(total - 100) > 1e-9:
            raise ValueError(f"group weights must sum to 100, got {total}")
        ids = [c.criterion_id for g in self.groups for c in g.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion_id must be unique across the rubric")
        return self

    def group_of(self, criterion_id: str) -> RubricGroup:
        for g in self.groups:
            if any(c.criterion_id == criterion_id for c in g.criteria):
                return g
        raise KeyError(criterion_id)
