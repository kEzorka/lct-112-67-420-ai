"""ИИ-руководитель (6.4): автомат диалога, сценарный fallback, учёт подтверждений C-05."""

from .ack import AckOutcome, AckSummary, summarize_acks
from .dialogue import Conversation, DialResult, Phase, Supervisor, Turn, TurnKind
from .templates import FallbackTemplates

__all__ = [
    "AckOutcome",
    "AckSummary",
    "Conversation",
    "DialResult",
    "FallbackTemplates",
    "Phase",
    "Supervisor",
    "Turn",
    "TurnKind",
    "summarize_acks",
]
