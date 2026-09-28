"""Адаптер llama.cpp: только локальный файл, версия модели, тайм-аут, грамматика JSON.

Настоящий `llama_cpp` в dev-окружении не ставится, поэтому модуль подменяется двойником
с тем же интерфейсом (`Llama` с потоковой генерацией, `LlamaGrammar`). Дымовой прогон на
настоящем llama.cpp — docs/ai/bench/llm-cpu.md.
"""

import json
import sys
import time
import types
from typing import ClassVar

import pytest

from dds_ai.contracts.events import ComponentName, FailureKind
from dds_ai.contracts.worker import Lane
from dds_ai.faults import ComponentFailure
from dds_ai.llm.llamacpp_provider import REPLY_SCHEMA, LlamaCppProvider, file_fingerprint
from dds_ai.worker import InferenceWorker

REPLY = json.dumps({"action": "listen", "text": "Слушаю."}, ensure_ascii=False)


class FakeLlama:
    """Потоковая генерация, как у `llama_cpp.Llama` при `stream=True`."""

    instances: ClassVar[list] = []

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.step_s = 0.0
        self.steps = 3
        self.yielded = 0
        self.closed = False
        self.calls: list[dict] = []
        FakeLlama.instances.append(self)

    def _stream(self, kwargs, wrap):
        self.calls.append(kwargs)
        assert kwargs["stream"] is True
        pieces = [REPLY[:5], REPLY[5:12], REPLY[12:]] + [""] * max(self.steps - 3, 0)

        def gen():
            try:
                for piece in pieces:
                    time.sleep(self.step_s)
                    self.yielded += 1
                    yield wrap(piece)
            finally:
                self.closed = True

        return gen()

    def create_chat_completion(self, messages, **kwargs):
        kwargs["messages"] = messages
        return self._stream(kwargs, lambda p: {"choices": [{"delta": {"content": p}}]})

    def create_completion(self, prompt, **kwargs):
        kwargs["prompt"] = prompt
        return self._stream(kwargs, lambda p: {"choices": [{"text": p}]})


class FakeGrammar:
    @classmethod
    def from_json_schema(cls, schema, verbose=True):
        g = cls()
        g.schema = json.loads(schema)
        return g


@pytest.fixture
def llama(monkeypatch):
    mod = types.ModuleType("llama_cpp")
    mod.Llama = FakeLlama
    mod.LlamaGrammar = FakeGrammar
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    FakeLlama.instances = []
    return mod


@pytest.fixture
def gguf(tmp_path):
    path = tmp_path / "candidate-q4.gguf"
    path.write_bytes(b"GGUF" + bytes(3 * (1 << 20)))
    return path


def test_missing_model_file_is_an_error_not_a_download(llama, tmp_path):
    with pytest.raises(FileNotFoundError):
        LlamaCppProvider(tmp_path / "absent.gguf")
    assert FakeLlama.instances == []


def test_model_ref_uses_explicit_version_or_file_fingerprint(llama, gguf):
    p = LlamaCppProvider(gguf, model_name="candidate", model_version="manifest-1")
    assert (p.model_ref.component, p.model_ref.model_name) == ("llm", "candidate")
    assert p.model_ref.model_version == "manifest-1"

    auto = LlamaCppProvider(gguf)
    assert auto.model_ref.model_name == "candidate-q4"
    assert auto.model_ref.model_version == file_fingerprint(gguf)
    gguf.write_bytes(gguf.read_bytes()[:-1] + b"\x01")  # подмена файла меняет версию
    assert file_fingerprint(gguf) != auto.model_ref.model_version


def test_runtime_is_configured_for_local_cpu_and_json_grammar(llama, gguf):
    p = LlamaCppProvider(gguf, n_threads=6, n_ctx=1024, seed=7)
    [inst] = FakeLlama.instances
    assert inst.model_path == str(gguf)
    assert inst.kwargs["n_threads"] == 6 and inst.kwargs["n_ctx"] == 1024
    assert inst.kwargs["verbose"] is False
    out = p.complete("промпт", max_tokens=32)
    assert json.loads(out) == {"action": "listen", "text": "Слушаю."}
    call = inst.calls[-1]
    assert call["messages"] == [{"role": "user", "content": "промпт"}]
    assert call["max_tokens"] == 32 and call["temperature"] == 0.0 and call["seed"] == 7
    assert call["grammar"].schema == REPLY_SCHEMA


def test_plain_completion_mode(llama, gguf):
    p = LlamaCppProvider(gguf, chat=False, json_grammar=False)
    p.complete("промпт", max_tokens=8)
    call = FakeLlama.instances[0].calls[-1]
    assert call["prompt"] == "промпт" and call["grammar"] is None


def test_generation_deadline_stops_decoding_and_raises_timeout(llama, gguf):
    p = LlamaCppProvider(gguf, timeout_s=0.05)
    inst = FakeLlama.instances[0]
    inst.step_s, inst.steps = 0.03, 50
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        p.complete("промпт", max_tokens=64)
    assert time.monotonic() - started < 0.5
    inst = FakeLlama.instances[0]
    assert inst.yielded < 10 and inst.closed  # поток закрыт, 50 шагов не дочитаны


def test_adapter_timeout_is_logged_by_worker_as_timeout(llama, gguf):
    p = LlamaCppProvider(gguf, timeout_s=0.05)
    FakeLlama.instances[0].step_s, FakeLlama.instances[0].steps = 0.03, 50
    w = InferenceWorker.from_profile("cpu")
    w.register(ComponentName.LLM, p)
    with pytest.raises(ComponentFailure) as exc:
        w.call(ComponentName.LLM, Lane.INTERACTIVE, lambda: p.complete("x", max_tokens=8))
    assert exc.value.kind is FailureKind.TIMEOUT
    assert w.injector.log[-1].kind is FailureKind.TIMEOUT
