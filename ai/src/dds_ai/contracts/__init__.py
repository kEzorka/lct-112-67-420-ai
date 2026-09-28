"""Контракты ИИ-контура. Все — черновики до согласования с бэкендом (B01–B04)."""

from pydantic import BaseModel, TypeAdapter

from .card import CardRevisionRecord, IncidentCard
from .criteria import CriterionResult
from .dialogue import Interlocutor, SupervisorReply
from .events import AttemptEvent, attempt_event_adapter
from .profile import Recommendation, SkillProfile
from .remarks import Remark
from .routing import RoutingDecision, RoutingRequest
from .rubric import Rubric
from .scoring import AttemptVersionSnapshot, ScoreSummary, ScoreVersion
from .worker import AdapterPolicy, FailureRecord, PreflightReport

CONTRACTS_VERSION = "0.1.0-draft"

# Имя файла JSON Schema в /contracts → модель.
SCHEMA_MODELS: dict[str, type[BaseModel] | TypeAdapter] = {
    "attempt-event": attempt_event_adapter,
    "incident-card": IncidentCard,
    "card-revision": CardRevisionRecord,
    "criterion-result": CriterionResult,
    "remark": Remark,
    "rubric": Rubric,
    "attempt-version-snapshot": AttemptVersionSnapshot,
    "score-summary": ScoreSummary,
    "score-version": ScoreVersion,
    "routing-request": RoutingRequest,
    "routing-decision": RoutingDecision,
    "interlocutor": Interlocutor,
    "supervisor-reply": SupervisorReply,
    "skill-profile": SkillProfile,
    "recommendation": Recommendation,
    "adapter-policy": AdapterPolicy,
    "failure-record": FailureRecord,
    "preflight-report": PreflightReport,
}

__all__ = ["CONTRACTS_VERSION", "SCHEMA_MODELS", "AttemptEvent"]
