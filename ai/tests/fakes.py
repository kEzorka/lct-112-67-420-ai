"""Тестовые двойники адаптеров моделей. Настоящих моделей в тестах нет."""

from __future__ import annotations

import json
from collections.abc import Callable

from dds_ai.contracts.common import Evidence, EvidenceKind, ModelRef
from dds_ai.contracts.criteria import CriterionResult, CriterionStatus
from dds_ai.llm import draft_of


class FakeTTS:
    model_ref = ModelRef(component="tts", model_name="fake-tts", model_version="0")

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[str] = []

    def synthesize(self, text: str, *, voice: str, intonation: str) -> bytes:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("synthesis crashed")
        return text.encode("utf-8")


class FakeSTT:
    model_ref = ModelRef(component="stt", model_name="fake-stt", model_version="0")

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[bytes] = []

    def transcribe(self, audio: bytes, *, hints=()):
        self.calls.append(audio)
        if self.fail:
            raise RuntimeError("transcription crashed")
        return []


def echo_draft(prompt: str) -> str:
    """Ответ «модели», которая формулирует реплику ровно как черновик."""
    action, draft = draft_of(prompt)
    return reply_json(action, draft)


def reply_json(action: str, text: str) -> str:
    return json.dumps({"action": action, "text": text}, ensure_ascii=False)


class FakeLLM:
    model_ref = ModelRef(component="llm", model_name="fake-llm", model_version="0")

    def __init__(self, answer: Callable[[str], str] | None = None):
        self.answer = answer or echo_draft
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.answer(prompt)


class StubJudge:
    """Оцениватель-заглушка: засчитывает семантические критерии (для проверки итогового балла)."""

    model_ref = ModelRef(component="semantic_judge", model_name="stub-judge", model_version="0")

    def judge(self, criterion_id: str, context: dict) -> CriterionResult:
        return CriterionResult(
            criterion_id=criterion_id,
            status=CriterionStatus.PASSED,
            value=1,
            evidence=(Evidence(kind=EvidenceKind.CARD_FIELD, ref="description"),),
            model_ref=self.model_ref,
            decided_by="model",
        )
