"""LLM-формулировка реплики руководителя внутри выбранного автоматом действия (M3, 6.4, C-04).

Автомат выбирает действие и черновик реплики из проверенного шаблона. LLM только формулирует
этот черновик. Её выход проверяет `validator.py`. При невалидном выходе делается один повтор,
затем возвращается «нет формулировки», и диалог берёт шаблонную реплику.

Каждый сбой идёт через `InferenceWorker.call()` (полоса `interactive`) и попадает в журнал:
`invalid_output` с кодом причины в `detail`, а также `timeout`, `overflow` и `error`.
Повтор делается только после `invalid_output`. После тайм-аута или ошибки сразу берётся
fallback, чтобы не удваивать задержку живого звонка.

LLM не меняет состояние диалога, карточку, маршрут, балл или статус попытки: `phrase()`
получает неизменяемый `PhraseRequest` и возвращает только текст (инвариант 2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

from ..contracts.common import ModelRef
from ..contracts.events import ComponentName, FailureKind
from ..contracts.worker import FailureRecord, Lane
from ..faults import ComponentFailure
from ..ports import LLMProvider
from ..worker import InferenceWorker
from .matching import digits
from .prompting import PhraseRequest, PromptTemplate
from .validator import InvalidReply, RejectReason, validate_output

MAX_REPLY_CHARS = 300
MAX_ATTEMPTS = 2  # один повтор после невалидного выхода
_REASONS = frozenset(r.value for r in RejectReason)

__all__ = [
    "MAX_REPLY_CHARS",
    "AttemptStages",
    "InvalidReply",
    "LLMPhraser",
    "PhraseOutcome",
    "validate_reply",
]


def validate_reply(text: str, allowed_context: str, *, max_chars: int = MAX_REPLY_CHARS) -> str:
    """Проверка шаблонной реплики: не пустая, ограничена по длине, без чисел вне контекста."""
    text = text.strip()
    if not text:
        raise InvalidReply(RejectReason.EMPTY)
    if len(text) > max_chars:
        raise InvalidReply(RejectReason.TOO_LONG, f"> {max_chars}")
    extra = digits(text) - digits(allowed_context)
    if extra:
        raise InvalidReply(RejectReason.NUMBER_OUTSIDE_CONTEXT, ",".join(sorted(extra)))
    return text


@dataclass(frozen=True)
class AttemptStages:
    """Замер одной попытки. None — стадия не выполнялась (сбой до неё)."""

    generation_s: float | None
    validation_s: float | None
    outcome: str  # "ok" или код сбоя: invalid_output:<причина>, timeout, overflow, error


@dataclass(frozen=True)
class PhraseOutcome:
    text: str | None  # None — формулировки нет, диалог берёт шаблон (C-04)
    attempts: tuple[AttemptStages, ...]
    failures: tuple[FailureRecord, ...]
    total_s: float  # от начала формулировки до готового текста или отказа

    @property
    def ok(self) -> bool:
        return self.text is not None


class LLMPhraser:
    def __init__(
        self,
        llm: LLMProvider,
        worker: InferenceWorker,
        *,
        role_title: str,
        template: PromptTemplate | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ):
        self.llm = llm
        self.worker = worker
        self.role_title = role_title
        self.template = template or PromptTemplate.load()
        self.max_attempts = max_attempts

    @property
    def model_ref(self) -> ModelRef:
        """Версия модели и промпта — основание каждой LLM-реплики (инвариант 6)."""
        return self.llm.model_ref.model_copy(update={"prompt_version": self.template.version})

    def phrase(
        self,
        request: PhraseRequest,
        allowed_context: str,
        *,
        attempt_id: UUID | None = None,
    ) -> PhraseOutcome:
        started = time.monotonic()
        log = self.worker.injector.log
        before = len(log)
        attempts: list[AttemptStages] = []
        reason: str | None = None
        text: str | None = None
        for _ in range(self.max_attempts):
            prompt = self.template.build(request, retry_reason=reason)
            stages: dict[str, float] = {}
            try:
                text = self.worker.call(
                    ComponentName.LLM,
                    Lane.INTERACTIVE,
                    lambda p=prompt, s=stages: self._generate(p, s, request, allowed_context),
                    attempt_id=attempt_id,
                )
            except ComponentFailure as exc:
                outcome = str(exc.kind)
                if exc.kind is FailureKind.INVALID_OUTPUT:
                    reason = exc.detail if exc.detail in _REASONS else "invalid_output"
                    outcome = f"{exc.kind}:{reason}"
                attempts.append(AttemptStages(stages.get("gen"), stages.get("val"), outcome))
                if exc.kind is not FailureKind.INVALID_OUTPUT:
                    break
                continue
            attempts.append(AttemptStages(stages.get("gen"), stages.get("val"), "ok"))
            break
        return PhraseOutcome(
            text=text,
            attempts=tuple(attempts),
            failures=tuple(log[before:]),
            total_s=time.monotonic() - started,
        )

    def _generate(
        self, prompt: str, stages: dict[str, float], request: PhraseRequest, context: str
    ) -> str:
        t0 = time.monotonic()
        try:
            raw = self.llm.complete(prompt, max_tokens=self.template.max_tokens)
        except TimeoutError:
            raise
        except Exception as exc:
            # Сбой runtime — `error`, даже если это ValueError (например, промпт длиннее
            # контекста): `invalid_output` — только про проверенный выход модели.
            raise ComponentFailure(
                ComponentName.LLM, FailureKind.ERROR, type(exc).__name__
            ) from exc
        t1 = time.monotonic()
        stages["gen"] = t1 - t0
        try:
            return validate_output(
                raw,
                action=request.action,
                draft=request.draft,
                context=context,
                max_chars=self.template.max_chars,
            )
        except InvalidReply as exc:
            # В журнал — только код причины, без текста модели и реплик (инвариант 9).
            raise ComponentFailure(
                ComponentName.LLM, FailureKind.INVALID_OUTPUT, str(exc.reason)
            ) from None
        finally:
            stages["val"] = time.monotonic() - t1
