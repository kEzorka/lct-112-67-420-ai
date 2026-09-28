"""Исполнитель одного адаптера: конечный тайм-аут, ограниченная очередь, лимит параллелизма.

Каждая пара (компонент, полоса) получает свой исполнитель со своими потоками, поэтому фоновая
нагрузка не занимает слоты интерактивной (6.1).

Тайм-аут считается от постановки в очередь: ожидание в очереди входит в бюджет ответа.
Поток, в котором модель зависла, прервать нельзя: он держит слот параллелизма до возврата,
и это видно в статусе (`running`) — так зависшая модель честно выглядит как занятая мощность.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TypeVar

from ..contracts.events import FailureKind
from ..contracts.worker import AdapterPolicy, ComponentHealth, ComponentStatus
from ..faults import ComponentFailure

T = TypeVar("T")


@dataclass
class ExecutorMetrics:
    completed: int = 0
    errors: int = 0
    timeouts: int = 0
    overflows: int = 0
    latencies_s: list[float] = field(default_factory=list)


class AdapterExecutor:
    def __init__(self, policy: AdapterPolicy):
        self.policy = policy
        self._pool = ThreadPoolExecutor(
            max_workers=policy.max_concurrency,
            thread_name_prefix=f"{policy.component}-{policy.lane}",
        )
        self._lock = threading.Lock()
        self._pending = 0  # в очереди + выполняются
        self._running = 0
        self.metrics = ExecutorMetrics()

    @property
    def capacity(self) -> int:
        return self.policy.max_concurrency + self.policy.max_queue

    @property
    def running(self) -> int:
        with self._lock:
            return self._running

    @property
    def queue_length(self) -> int:
        with self._lock:
            return self._pending - self._running

    @property
    def overloaded(self) -> bool:
        with self._lock:
            return self._pending >= self.capacity

    def run(self, fn: Callable[[], T]) -> T:
        """Выполнить вызов в пределах политики.

        Переполнение → ComponentFailure(overflow); превышение тайм-аута → TimeoutError
        (FaultInjector превращает его в запись о сбое `timeout`).
        """
        with self._lock:
            if self._pending >= self.capacity:
                self.metrics.overflows += 1
                raise ComponentFailure(
                    self.policy.component,
                    FailureKind.OVERFLOW,
                    f"{self.policy.lane} queue full ({self.capacity})",
                )
            self._pending += 1

        started = time.monotonic()
        try:
            future = self._pool.submit(self._task, fn)
        except BaseException:
            self._release(None)
            raise
        future.add_done_callback(self._release)
        try:
            result = future.result(timeout=self.policy.timeout_s)
        except TimeoutError:
            future.cancel()  # из очереди снимается; выполняющийся вызов держит слот до возврата
            with self._lock:
                self.metrics.timeouts += 1
            raise TimeoutError(
                f"{self.policy.component}/{self.policy.lane} exceeded {self.policy.timeout_s}s"
            ) from None
        except BaseException:
            with self._lock:
                self.metrics.errors += 1
            raise
        with self._lock:
            self.metrics.completed += 1
            self.metrics.latencies_s.append(time.monotonic() - started)
        return result

    def _task(self, fn: Callable[[], T]) -> T:
        with self._lock:
            self._running += 1
        try:
            return fn()
        finally:
            with self._lock:
                self._running -= 1

    def _release(self, _future: Future | None) -> None:
        with self._lock:
            self._pending -= 1

    def status(self, *, available: bool, model_version: str | None) -> ComponentStatus:
        if not available:
            health = ComponentHealth.DOWN
        elif self.overloaded:
            health = ComponentHealth.OVERLOADED
        else:
            health = ComponentHealth.UP
        return ComponentStatus(
            component=self.policy.component,
            lane=self.policy.lane,
            health=health,
            model_version=model_version,
            queue_length=self.queue_length,
            running=self.running,
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
