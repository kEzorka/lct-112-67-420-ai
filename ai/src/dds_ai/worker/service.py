"""Inference-worker: реестр адаптеров, исполнители по полосам, статус и preflight (6.1, C-04).

Все вызовы моделей идут через `InferenceWorker.call()` → `FaultInjector.call()` → исполнитель,
поэтому тайм-аут, переполнение и ошибка модели одинаково попадают в журнал сбоев.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

from ..contracts.events import ComponentName, FailureKind
from ..contracts.worker import (
    AdapterPolicy,
    ComponentHealth,
    ComponentStatus,
    FailureRecord,
    Lane,
    PreflightReport,
)
from ..faults import FaultInjector
from .executor import AdapterExecutor

T = TypeVar("T")

CONFIG = Path(__file__).resolve().parents[3] / "config"


def load_policies(profile: str = "cpu") -> tuple[str, tuple[AdapterPolicy, ...]]:
    """Политики адаптеров профиля из `ai/config/worker.<profile>.json`."""
    data = json.loads((CONFIG / f"worker.{profile}.json").read_text("utf-8"))
    return data["profile"], tuple(AdapterPolicy.model_validate(p) for p in data["policies"])


class NotConfigured(LookupError):
    pass


class InferenceWorker:
    def __init__(
        self,
        policies: Iterable[AdapterPolicy],
        *,
        injector: FaultInjector | None = None,
        profile: str = "cpu",
    ):
        self.profile = profile
        self.injector = injector or FaultInjector()
        self._executors: dict[tuple[ComponentName, Lane], AdapterExecutor] = {}
        for p in policies:
            key = (p.component, p.lane)
            if key in self._executors:
                raise ValueError(f"duplicate policy for {key}")
            self._executors[key] = AdapterExecutor(p)
        self._adapters: dict[ComponentName, Any] = {}
        self._disabled: set[ComponentName] = set()

    @classmethod
    def from_profile(cls, profile: str = "cpu", **kw: Any) -> InferenceWorker:
        name, policies = load_policies(profile)
        return cls(policies, profile=name, **kw)

    # --- адаптеры --------------------------------------------------------------

    def register(self, component: ComponentName, adapter: Any) -> None:
        """Адаптер модели (STTProvider, TTSProvider, ...). Его model_ref идёт в статус."""
        self._adapters[component] = adapter

    def adapter(self, component: ComponentName) -> Any:
        return self._adapters.get(component)

    def set_available(self, component: ComponentName, available: bool) -> None:
        """Административное отключение компонента (C-04: «отключить по отдельности»)."""
        if available:
            self._disabled.discard(component)
        else:
            self._disabled.add(component)

    def available(self, component: ComponentName) -> bool:
        return (
            component in self._adapters
            and component not in self._disabled
            and self.injector.fault_for(component) is None
        )

    # --- вызов -----------------------------------------------------------------

    def executor(self, component: ComponentName, lane: Lane) -> AdapterExecutor:
        try:
            return self._executors[(component, lane)]
        except KeyError:
            raise NotConfigured(f"no policy for {component}/{lane}") from None

    def call(
        self,
        component: ComponentName,
        lane: Lane,
        fn: Callable[[], T],
        *,
        attempt_id: UUID | None = None,
        affected_evidence: tuple[str, ...] = (),
    ) -> T:
        executor = self.executor(component, lane)

        def guarded() -> T:
            if component in self._disabled:
                raise RuntimeError("component disabled by administrator")
            return executor.run(fn)

        return self.injector.call(
            component, guarded, attempt_id=attempt_id, affected_evidence=affected_evidence
        )

    # --- статус и preflight ------------------------------------------------------

    def status(self) -> tuple[ComponentStatus, ...]:
        """Статус всех исполнителей для администратора (D-058)."""
        out = []
        for (component, _lane), ex in self._executors.items():
            adapter = self._adapters.get(component)
            ref = getattr(adapter, "model_ref", None)
            out.append(
                ex.status(
                    available=self.available(component),
                    model_version=None if ref is None else f"{ref.model_name}@{ref.model_version}",
                )
            )
        return tuple(out)

    def preflight(
        self,
        required: Iterable[tuple[ComponentName, Lane]],
        *,
        voice_path_ok: bool,
    ) -> PreflightReport:
        """Готовность к старту попытки: нужные адаптеры и голосовой тракт (C-04)."""
        by_key = {(s.component, s.lane): s for s in self.status()}
        components = []
        for component, lane in required:
            s = by_key.get((component, lane))
            if s is None:
                s = ComponentStatus(component=component, lane=lane, health=ComponentHealth.DOWN)
            components.append(s)
        return PreflightReport(
            profile=self.profile, components=tuple(components), voice_path_ok=voice_path_ok
        )

    def preflight_failures(self, report: PreflightReport, *, at: datetime) -> list[FailureRecord]:
        """Компоненты preflight не в состоянии `up` — сбой пишет воркер (решение D-3), а не
        клиент: воркер отвечает за preflight (6.1), поэтому и за событие его провала."""
        out = []
        for c in report.components:
            if c.health is ComponentHealth.UP:
                continue
            overloaded = c.health is ComponentHealth.OVERLOADED
            kind = FailureKind.OVERFLOW if overloaded else FailureKind.ERROR
            out.append(
                FailureRecord(
                    component=c.component, kind=kind, at=at, detail=f"preflight: {c.health.value}"
                )
            )
        return out

    def shutdown(self) -> None:
        for ex in self._executors.values():
            ex.shutdown()
