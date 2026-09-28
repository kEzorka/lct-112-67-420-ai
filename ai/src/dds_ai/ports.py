"""Порты: адаптеры моделей (владеет ИИ-контур) и соседние модули (моки до появления бэкенда).

Модели вызываются только через эти протоколы; замена модели не меняет контракты (6.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from .contracts.card import CardRevisionRecord, IncidentCard
from .contracts.common import ModelRef
from .contracts.criteria import CriterionResult
from .contracts.events import AttemptEvent
from .contracts.routing import RoutingDecision, RoutingRequest


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_ms: int
    end_ms: int
    final: bool
    confidence: float | None = None  # только если модель его отдаёт (инвариант 5)


# --- адаптеры моделей -------------------------------------------------------


class STTProvider(Protocol):
    model_ref: ModelRef

    def transcribe(self, audio: bytes, *, hints: Sequence[str] = ()) -> list[TranscriptSegment]: ...


class TTSProvider(Protocol):
    model_ref: ModelRef

    def synthesize(self, text: str, *, voice: str, intonation: str) -> bytes: ...


class LLMProvider(Protocol):
    model_ref: ModelRef

    def complete(self, prompt: str, *, max_tokens: int) -> str: ...


class SemanticJudge(Protocol):
    model_ref: ModelRef

    def judge(self, criterion_id: str, context: dict) -> CriterionResult: ...


class Recommender(Protocol):
    model_ref: ModelRef

    def rank(self, pool_task_ids: Sequence[str], features: dict) -> list[str]: ...


class Embedder(Protocol):
    model_ref: ModelRef

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# --- соседние модули (черновики контрактов) ---------------------------------


class RoutingEngine(Protocol):
    def route(self, request: RoutingRequest) -> RoutingDecision: ...


class CardStore(Protocol):
    def put_source(self, card: IncidentCard) -> None: ...

    def get_source(self, card_id: UUID) -> IncidentCard: ...

    def add_revision(self, revision: CardRevisionRecord) -> None: ...

    def revisions(self, card_id: UUID) -> list[CardRevisionRecord]: ...


class MediaTransport(Protocol):
    def play(
        self, *, attempt_id: UUID, call_id: UUID, ack_id: UUID, utterance_id: UUID, audio: bytes
    ) -> list[AttemptEvent]: ...
