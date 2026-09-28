"""Inference-worker (6.1, C-04): тайм-аут, очередь, параллелизм, полосы, статус, preflight."""

import threading
import time

import pytest

from dds_ai.contracts.events import ComponentName, FailureKind
from dds_ai.contracts.worker import AdapterPolicy, ComponentHealth, Lane
from dds_ai.faults import ComponentFailure, FaultInjector
from dds_ai.worker import AdapterExecutor, InferenceWorker, NotConfigured, load_policies

from .fakes import FakeLLM, FakeTTS

LLM = ComponentName.LLM


def policy(lane=Lane.INTERACTIVE, *, timeout=1.0, queue=1, conc=1, component=LLM):
    return AdapterPolicy(
        component=component,
        lane=lane,
        timeout_s=timeout,
        max_queue=queue,
        max_concurrency=conc,
        failure_policy_version="test-1",
    )


@pytest.fixture
def gate():
    """Блокирует «модель», пока тест не отпустит; освобождается в конце теста."""
    ev = threading.Event()
    yield ev
    ev.set()


@pytest.fixture
def worker():
    w = InferenceWorker(
        [policy(Lane.INTERACTIVE, conc=1, queue=1), policy(Lane.BACKGROUND, conc=1, queue=2)]
    )
    w.register(LLM, FakeLLM())
    yield w
    w.shutdown()


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


def in_background(fn):
    def quiet():
        try:
            fn()
        except ComponentFailure:
            pass  # вызовы, снятые при остановке исполнителя в конце теста

    t = threading.Thread(target=quiet, daemon=True)
    t.start()
    return t


def test_timeout_is_finite_and_logged(gate):
    w = InferenceWorker([policy(timeout=0.05)])
    started = time.monotonic()
    with pytest.raises(ComponentFailure) as exc:
        w.call(LLM, Lane.INTERACTIVE, lambda: gate.wait(5))
    assert time.monotonic() - started < 1.0
    assert exc.value.kind is FailureKind.TIMEOUT
    assert [r.kind for r in w.injector.log] == [FailureKind.TIMEOUT]
    assert w.executor(LLM, Lane.INTERACTIVE).metrics.timeouts == 1
    w.shutdown()


def test_queue_overflow_is_explicit_and_logged(worker, gate):
    ex = worker.executor(LLM, Lane.INTERACTIVE)  # 1 выполняется + 1 в очереди
    errors: list[ComponentFailure] = []

    def blocked():
        try:
            worker.call(LLM, Lane.INTERACTIVE, lambda: gate.wait(5))
        except ComponentFailure as e:  # pragma: no cover - не должно случиться
            errors.append(e)

    in_background(blocked)
    wait_until(lambda: ex.running == 1)
    in_background(blocked)
    wait_until(lambda: ex.queue_length == 1)

    assert ex.overloaded
    status = {(s.component, s.lane): s for s in worker.status()}[(LLM, Lane.INTERACTIVE)]
    assert status.health is ComponentHealth.OVERLOADED
    assert (status.running, status.queue_length) == (1, 1)

    with pytest.raises(ComponentFailure) as exc:
        worker.call(LLM, Lane.INTERACTIVE, lambda: "never runs")
    assert exc.value.kind is FailureKind.OVERFLOW
    assert worker.injector.log[-1].kind is FailureKind.OVERFLOW
    gate.set()
    wait_until(lambda: not ex.overloaded and ex.running == 0)
    assert not errors


def test_background_load_does_not_block_interactive(worker, gate):
    bg = worker.executor(LLM, Lane.BACKGROUND)  # 1 + 2 в очереди — полностью занят
    for _ in range(3):
        in_background(lambda: worker.call(LLM, Lane.BACKGROUND, lambda: gate.wait(5)))
    wait_until(lambda: bg.overloaded)
    with pytest.raises(ComponentFailure):
        worker.call(LLM, Lane.BACKGROUND, lambda: "x")

    started = time.monotonic()
    assert worker.call(LLM, Lane.INTERACTIVE, lambda: "ответ") == "ответ"
    assert time.monotonic() - started < 0.5
    health = {(s.component, s.lane): s.health for s in worker.status()}
    assert health[(LLM, Lane.INTERACTIVE)] is ComponentHealth.UP
    assert health[(LLM, Lane.BACKGROUND)] is ComponentHealth.OVERLOADED


def test_concurrency_limit_is_respected(gate):
    ex = AdapterExecutor(policy(conc=2, queue=10, timeout=5))
    peak = 0
    lock = threading.Lock()
    active = 0

    def job():
        nonlocal peak, active
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return 1

    threads = [in_background(lambda: ex.run(job)) for _ in range(6)]
    for t in threads:
        t.join(2)
    assert peak == 2
    assert ex.metrics.completed == 6
    ex.shutdown()


def test_model_error_goes_through_fault_injector():
    w = InferenceWorker([policy()])

    def boom():
        raise RuntimeError("crash")

    with pytest.raises(ComponentFailure) as exc:
        w.call(LLM, Lane.INTERACTIVE, boom)
    assert exc.value.kind is FailureKind.ERROR
    assert w.injector.log[-1].component is LLM


def test_injected_fault_applies_to_worker_calls():
    inj = FaultInjector()
    w = InferenceWorker([policy()], injector=inj)
    inj.inject(LLM, FailureKind.TIMEOUT)
    with pytest.raises(ComponentFailure) as exc:
        w.call(LLM, Lane.INTERACTIVE, lambda: "x")
    assert exc.value.kind is FailureKind.TIMEOUT


def test_unconfigured_lane_is_rejected():
    w = InferenceWorker([policy(Lane.INTERACTIVE)])
    with pytest.raises(NotConfigured):
        w.call(LLM, Lane.BACKGROUND, lambda: "x")


def test_disabled_component_fails_and_is_down():
    w = InferenceWorker([policy()])
    w.register(LLM, FakeLLM())
    w.set_available(LLM, False)
    with pytest.raises(ComponentFailure):
        w.call(LLM, Lane.INTERACTIVE, lambda: "x")
    assert w.status()[0].health is ComponentHealth.DOWN
    w.set_available(LLM, True)
    assert w.call(LLM, Lane.INTERACTIVE, lambda: "x") == "x"


def test_preflight_reports_missing_models_and_voice_path():
    w = InferenceWorker.from_profile("cpu")
    w.register(LLM, FakeLLM())
    w.register(ComponentName.TTS, FakeTTS())
    required = [(LLM, Lane.INTERACTIVE), (ComponentName.TTS, Lane.INTERACTIVE)]
    assert w.preflight(required, voice_path_ok=True).ready
    assert not w.preflight(required, voice_path_ok=False).ready

    stt = w.preflight([(ComponentName.STT, Lane.INTERACTIVE)], voice_path_ok=True)
    assert not stt.ready and stt.components[0].health is ComponentHealth.DOWN  # не загружен

    w.injector.inject(ComponentName.TTS, FailureKind.ERROR)
    report = w.preflight(required, voice_path_ok=True)
    assert not report.ready
    assert {c.component: c.health for c in report.components}[ComponentName.TTS] is (
        ComponentHealth.DOWN
    )
    assert report.components[0].model_version == "fake-llm@0"


def test_cpu_profile_has_finite_policies_for_every_model_component():
    profile, policies = load_policies("cpu")
    assert profile == "cpu"
    covered = {p.component for p in policies}
    assert covered == set(ComponentName)
    assert all(p.timeout_s > 0 and p.max_queue >= 1 for p in policies)
    lanes = {(p.component, p.lane) for p in policies}
    for component in (LLM, ComponentName.STT, ComponentName.TTS):
        assert (component, Lane.INTERACTIVE) in lanes
