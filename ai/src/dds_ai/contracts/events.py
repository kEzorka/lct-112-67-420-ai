"""События попытки: append-only журнал (Miro 05, C-01, C-05).

Черновик контракта: журнал хранит бэкенд, ИИ-контур пишет только события со source=ai_worker
или media и читает остальные. `played_audio_ack` с доски — это `ack.played_by_client`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, TypeAdapter

from .common import Contract, ModelRef, NonEmptyStr
from .dialogue import ScenarioCallState


class EventSource(StrEnum):
    CLIENT = "client"
    BACKEND = "backend"
    AI_WORKER = "ai_worker"
    MEDIA = "media"


class DispatcherDecision(StrEnum):
    RESPOND = "respond"
    REFUSE = "refuse"
    REDIRECT = "redirect"


class CallState(StrEnum):
    ACTIVE = "active"
    MUTED = "muted"
    HOLD = "hold"
    ENDED = "ended"


class Speaker(StrEnum):
    DISPATCHER = "dispatcher"
    SUPERVISOR = "supervisor"


class ComponentName(StrEnum):
    STT = "stt"
    TTS = "tts"
    LLM = "llm"
    SEMANTIC_JUDGE = "semantic_judge"
    RECOMMENDER = "recommender"
    EMBEDDER = "embedder"


class FailureKind(StrEnum):
    ERROR = "error"
    TIMEOUT = "timeout"
    OVERFLOW = "overflow"
    INVALID_OUTPUT = "invalid_output"


class _Event(Contract):
    event_id: UUID
    attempt_id: UUID
    seq: int = Field(ge=0, description="Порядковый номер в попытке, назначает бэкенд")
    server_ts: AwareDatetime
    client_ts: AwareDatetime | None = None
    source: EventSource


class NotificationShown(_Event):
    type: Literal["notification_shown"] = "notification_shown"


class CardOpened(_Event):
    type: Literal["card_opened"] = "card_opened"
    card_id: UUID


class CardRevision(_Event):
    type: Literal["card_revision"] = "card_revision"
    card_id: UUID
    revision: int = Field(ge=1)
    changed_fields: tuple[NonEmptyStr, ...] = Field(min_length=1)


class SelectedAction(_Event):
    type: Literal["selected_action"] = "selected_action"
    decision: DispatcherDecision


class SubmitClicked(_Event):
    type: Literal["submit_clicked"] = "submit_clicked"


class SubmitAccepted(_Event):
    """Подтверждённая сервером сдача: точка attempt_submitted_at по C-01."""

    type: Literal["submit_accepted"] = "submit_accepted"
    incomplete: bool = Field(
        default=False, description="«Завершить с невыполненными шагами» (C-02): не успех"
    )
    missing_steps: tuple[NonEmptyStr, ...] = Field(
        default=(), description="Наблюдаемые пропуски; фиктивные события не создаются (C-02)"
    )


class Timeout(_Event):
    """Тайм-аут — только событие: не создаёт действие, звонок или сдачу."""

    type: Literal["timeout"] = "timeout"
    limit: Literal["open", "processing"]


class CallStateChanged(_Event):
    type: Literal["call_state_changed"] = "call_state_changed"
    call_id: UUID
    state: CallState


class ScenarioCallStateChanged(_Event):
    """Сценарное состояние вызова (занято, не отвечает, обрыв) — не технический сбой."""

    type: Literal["scenario_call_state"] = "scenario_call_state"
    call_id: UUID | None = Field(default=None, description="None — текстовый режим")
    role_id: NonEmptyStr
    state: ScenarioCallState


class Utterance(_Event):
    type: Literal["utterance"] = "utterance"
    call_id: UUID | None = Field(default=None, description="None — текстовый режим")
    utterance_id: UUID
    speaker: Speaker
    text: str
    final: bool = True
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    model_ref: ModelRef | None = None
    fallback: bool = Field(
        default=False, description="Реплика из детерминированного шаблона (C-04)"
    )


class _Ack(_Event):
    ack_id: UUID
    call_id: UUID
    utterance_id: UUID


class AckGenerated(_Ack):
    type: Literal["ack.generated"] = "ack.generated"


class AckSentToMedia(_Ack):
    type: Literal["ack.sent_to_media"] = "ack.sent_to_media"


class AckPlayedByClient(_Ack):
    type: Literal["ack.played_by_client"] = "ack.played_by_client"


class AckDeliveryUnconfirmed(_Ack):
    type: Literal["ack.delivery_unconfirmed"] = "ack.delivery_unconfirmed"
    reason: NonEmptyStr


class TextDelivered(_Event):
    """Доставка текста в аварийном текстовом режиме — не голосовое подтверждение."""

    type: Literal["text_delivered"] = "text_delivered"
    utterance_id: UUID
    ack_id: UUID | None = Field(
        default=None, description="Подтверждение в текстовом режиме; не голосовое (C-05)"
    )


class HintShown(_Event):
    type: Literal["hint_shown"] = "hint_shown"
    hint_id: UUID
    text: str


class HelpRequested(_Event):
    type: Literal["help_requested"] = "help_requested"


class ModelFailure(_Event):
    type: Literal["model_failure"] = "model_failure"
    component: ComponentName
    kind: FailureKind
    detail: str | None = None
    recovered: bool = Field(
        default=False,
        description="Выход модели отклонён, но повтор прошёл: не техническое нарушение",
    )


AttemptEvent = Annotated[
    NotificationShown
    | CardOpened
    | CardRevision
    | SelectedAction
    | SubmitClicked
    | SubmitAccepted
    | Timeout
    | CallStateChanged
    | ScenarioCallStateChanged
    | Utterance
    | AckGenerated
    | AckSentToMedia
    | AckPlayedByClient
    | AckDeliveryUnconfirmed
    | TextDelivered
    | HintShown
    | HelpRequested
    | ModelFailure,
    Field(discriminator="type"),
]

attempt_event_adapter: TypeAdapter[AttemptEvent] = TypeAdapter(AttemptEvent)


def parse_event(data: dict | str | bytes) -> AttemptEvent:
    if isinstance(data, dict):
        return attempt_event_adapter.validate_python(data)
    return attempt_event_adapter.validate_json(data)
