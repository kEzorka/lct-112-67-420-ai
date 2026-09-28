"""Адаптер LLMProvider на llama.cpp (`llama-cpp-python`, GGUF, CPU).

Extras: `pip install "dds-ai[llm-llamacpp]"`. Модель — только локальный файл GGUF из пакета
поставки. Адаптер ничего не скачивает (инвариант 8, C-10): путь не к существующему файлу —
ошибка при создании.

Тайм-аут. Исполнитель воркера снимает вызов с ожидания по `timeout_s` политики, но поток
llama.cpp этим не прерывается. Поэтому у адаптера свой срок генерации. Генерация идёт
потоком токенов. Когда срок истекает, адаптер перестаёт читать поток: генерация
останавливается на следующем токене, а адаптер бросает `TimeoutError`, который воркер пишет
как `timeout`. Обработку промпта до первого токена так прервать нельзя. Этот отрезок
ограничен длиной промпта: история в нём ограничена (`history_turns`).

Принудительный токен конца через обработчик логитов не годится: с грамматикой JSON
llama.cpp аварийно завершает процесс, если грамматика не допускает конец
(«Unexpected empty grammar stack»). Это найдено дымовым прогоном на крошечной модели
(`ai/bench/make_tiny_gguf.py`).

Формат выхода. По умолчанию генерация ограничена грамматикой JSON-схемы
`{"action", "text"}`. Грамматика задаёт только форму ответа. Содержание по-прежнему
проверяет `supervisor/validator.py`.

Версия модели. Берётся явная `model_version` из манифеста поставки. Если её нет —
отпечаток файла: размер и sha256 первого и последнего мебибайта. Это не полный хеш,
а быстрый признак подмены файла; для аудита нужен полный хеш в манифесте.

В этом окружении с настоящими весами не запускался: huggingface.co закрыт сетевой
политикой (docs/ai/bench/llm-cpu.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from ..contracts.common import ModelRef

_CHUNK = 1 << 20

REPLY_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string"}, "text": {"type": "string"}},
    "required": ["action", "text"],
    "additionalProperties": False,
}


def file_fingerprint(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(_CHUNK))
        if size > _CHUNK:
            f.seek(max(size - _CHUNK, _CHUNK))
            h.update(f.read(_CHUNK))
    return f"size{size}-sha256p-{h.hexdigest()[:16]}"


class LlamaCppProvider:
    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        model_name: str | None = None,
        model_version: str | None = None,
        timeout_s: float = 8.0,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        json_grammar: bool = True,
        chat: bool = True,
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model not found locally: {path} (no downloads)")
        try:
            import llama_cpp
        except ImportError as exc:  # pragma: no cover - зависит от опциональной зависимости
            raise ImportError(
                'llama-cpp-python не установлен: pip install "dds-ai[llm-llamacpp]"'
            ) from exc
        self._lib = llama_cpp
        self._llm = llama_cpp.Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            seed=seed,
            verbose=False,
        )
        self._grammar = (
            llama_cpp.LlamaGrammar.from_json_schema(json.dumps(REPLY_SCHEMA), verbose=False)
            if json_grammar
            else None
        )
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.seed = seed
        self.chat = chat
        self.model_ref = ModelRef(
            component="llm",
            model_name=model_name or path.stem,
            model_version=model_version or file_fingerprint(path),
        )

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        deadline = time.monotonic() + self.timeout_s
        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "grammar": self._grammar,
            "stream": True,
        }
        if self.chat:
            stream = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}], **kwargs
            )
            pieces = (c["choices"][0]["delta"].get("content") or "" for c in stream)
        else:
            stream = self._llm.create_completion(prompt, **kwargs)
            pieces = (c["choices"][0]["text"] for c in stream)
        out: list[str] = []
        try:
            for piece in pieces:
                out.append(piece)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"llama.cpp generation exceeded {self.timeout_s}s")
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()  # остановить генерацию, а не дочитывать её
        return "".join(out)
