"""Контракты ИИ-контура. Все — черновики до согласования с бэкендом (B01–B04)."""

from pydantic import BaseModel, TypeAdapter

from .analytics import GroupErrorReport
from .card import CardRevisionRecord, IncidentCard
from .criteria import CriterionResult
from .dialogue import Interlocutor, SupervisorReply
from .difficulty import DifficultyProposal, DifficultyScore, DifficultyVector, DifficultyWeights
from .events import AttemptEvent, attempt_event_adapter
from .knowledge import CorpusVersion, KnowledgeFragment
from .nlp import NlpExtraction
from .profile import Recommendation, SkillProfile, TaskPool
from .remarks import ErrorExplanation, Remark
from .routing import RoutingDecision, RoutingRequest
from .rubric import Rubric
from .scenario import Scenario
from .scenario_lifecycle import ScenarioDraftRecord
from .scoring import AttemptVersionSnapshot, ScoreSummary, ScoreVersion
from .worker import AdapterPolicy, FailureRecord, PreflightReport

CONTRACTS_VERSION = "0.4.0-draft"

# Имя файла JSON Schema в /contracts → модель.
SCHEMA_MODELS: dict[str, type[BaseModel] | TypeAdapter] = {
    "attempt-event": attempt_event_adapter,
    "incident-card": IncidentCard,
    "card-revision": CardRevisionRecord,
    "criterion-result": CriterionResult,
    "remark": Remark,
    "error-explanation": ErrorExplanation,
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
    "corpus-version": CorpusVersion,
    "knowledge-fragment": KnowledgeFragment,
    "nlp-extraction": NlpExtraction,
    "scenario": Scenario,
    "scenario-draft-record": ScenarioDraftRecord,
    "difficulty-vector": DifficultyVector,
    "difficulty-weights": DifficultyWeights,
    "difficulty-score": DifficultyScore,
    "difficulty-proposal": DifficultyProposal,
    "task-pool": TaskPool,
    "group-error-report": GroupErrorReport,
}

__all__ = ["CONTRACTS_VERSION", "SCHEMA_MODELS", "AttemptEvent"]
