"""Вызов семантического оценивателя с деградацией: сбой модели → not_checked, не 0 (инвариант 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from .contracts.criteria import CriterionResult, CriterionStatus
from .contracts.events import ComponentName
from .contracts.worker import Lane
from .faults import ComponentFailure, FaultInjector, InvalidOutput
from .ports import SemanticJudge

if TYPE_CHECKING:
    from .worker import InferenceWorker


def judge_criterion(
    judge: SemanticJudge,
    injector: FaultInjector,
    criterion_id: str,
    context: dict | None = None,
    *,
    attempt_id: UUID | None = None,
    worker: InferenceWorker | None = None,
) -> CriterionResult:
    """С `worker` вызов идёт в фоновую полосу исполнителя (тайм-аут, очередь, параллелизм)."""

    def run() -> CriterionResult:
        result = judge.judge(criterion_id, context or {})
        if not isinstance(result, CriterionResult):
            result = CriterionResult.model_validate(result)
        if result.criterion_id != criterion_id:
            raise InvalidOutput("judge answered for a different criterion")
        return result

    try:
        if worker is not None:
            return worker.call(
                ComponentName.SEMANTIC_JUDGE, Lane.BACKGROUND, run, attempt_id=attempt_id
            )
        return injector.call(ComponentName.SEMANTIC_JUDGE, run, attempt_id=attempt_id)
    except ComponentFailure as exc:
        return CriterionResult(
            criterion_id=criterion_id,
            status=CriterionStatus.NOT_CHECKED,
            explanation=(
                f"Не проверено: сбой оценивателя ({exc.kind}). "
                "Ожидает повторного прогона или эксперта."
            ),
        )
