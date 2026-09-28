"""Инъекция отказов и политика деградации (C-04).

Любой вызов модели идёт через FaultInjector.call(): в тестах можно отключить компонент,
вызвать тайм-аут, переполнение очереди или невалидный выход. Каждый сбой попадает в журнал.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar
from uuid import UUID

from pydantic import ValidationError

from .contracts.events import ComponentName, FailureKind
from .contracts.worker import FailureRecord

T = TypeVar("T")


class InvalidOutput(ValueError):
    """Выход модели не прошёл проверку. Только он (и ValidationError) даёт `invalid_output`;
    прочий ValueError — ошибка runtime (`error`)."""


class ComponentFailure(Exception):
    def __init__(self, component: ComponentName, kind: FailureKind, detail: str | None = None):
        super().__init__(f"{component}: {kind}" + (f" ({detail})" if detail else ""))
        self.component = component
        self.kind = kind
        self.detail = detail


class Degradation(StrEnum):
    """Поведение при отказе компонента — таблица C-04 из промпта (6.1)."""

    SCRIPTED_FALLBACK = "scripted_fallback"
    TEXT_MODE = "text_mode"
    NOT_CHECKED = "not_checked"
    TEACHER_SEQUENCE = "teacher_sequence"
    GENERATORS_WITHOUT_RAG = "generators_without_rag"


DEGRADATION: dict[ComponentName, Degradation] = {
    ComponentName.LLM: Degradation.SCRIPTED_FALLBACK,
    ComponentName.STT: Degradation.TEXT_MODE,
    ComponentName.TTS: Degradation.TEXT_MODE,
    ComponentName.SEMANTIC_JUDGE: Degradation.NOT_CHECKED,
    ComponentName.RECOMMENDER: Degradation.TEACHER_SEQUENCE,
    ComponentName.EMBEDDER: Degradation.GENERATORS_WITHOUT_RAG,
}


class FaultInjector:
    def __init__(self, clock: Callable[[], datetime] | None = None):
        self._faults: dict[ComponentName, FailureKind] = {}
        self._clock = clock or (lambda: datetime.now(UTC))
        self.log: list[FailureRecord] = []

    def inject(self, component: ComponentName, kind: FailureKind) -> None:
        self._faults[component] = kind

    def fault_for(self, component: ComponentName) -> FailureKind | None:
        """Внедрённый отказ компонента (для preflight и статуса)."""
        return self._faults.get(component)

    def clear(self, component: ComponentName | None = None) -> None:
        if component is None:
            self._faults.clear()
        else:
            self._faults.pop(component, None)

    def call(
        self,
        component: ComponentName,
        fn: Callable[[], T],
        *,
        attempt_id: UUID | None = None,
        affected_evidence: tuple[str, ...] = (),
    ) -> T:
        """Выполнить вызов модели; любой сбой — ComponentFailure и запись в журнал."""
        try:
            kind = self._faults.get(component)
            if kind is FailureKind.INVALID_OUTPUT:
                fn()
                raise ComponentFailure(component, kind, "injected invalid output")
            if kind is not None:
                raise ComponentFailure(component, kind, "injected")
            return fn()
        except ComponentFailure as exc:
            self._record(exc, attempt_id, affected_evidence)
            raise
        except TimeoutError as exc:
            failure = ComponentFailure(component, FailureKind.TIMEOUT, str(exc) or None)
            self._record(failure, attempt_id, affected_evidence)
            raise failure from exc
        except (ValidationError, InvalidOutput) as exc:
            failure = ComponentFailure(component, FailureKind.INVALID_OUTPUT, type(exc).__name__)
            self._record(failure, attempt_id, affected_evidence)
            raise failure from exc
        except Exception as exc:
            failure = ComponentFailure(component, FailureKind.ERROR, type(exc).__name__)
            self._record(failure, attempt_id, affected_evidence)
            raise failure from exc

    def _record(self, exc: ComponentFailure, attempt_id: UUID | None, evidence: tuple[str, ...]):
        self.log.append(
            FailureRecord(
                component=exc.component,
                kind=exc.kind,
                at=self._clock(),
                attempt_id=attempt_id,
                affected_evidence=evidence,
                detail=exc.detail,
            )
        )
