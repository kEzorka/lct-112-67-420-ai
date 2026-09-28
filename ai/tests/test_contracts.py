"""Прочие контракты: реплики, профиль, рекомендации, версии оценки, замечания, JSON Schema."""

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from dds_ai.contracts.common import Evidence, EvidenceKind, ModelRef, VersionRef
from dds_ai.contracts.dialogue import Interlocutor, ReplyMode, SupervisorAction, SupervisorReply
from dds_ai.contracts.profile import (
    GroupLevel,
    ProfileStatus,
    Recommendation,
    RecommendationStatus,
    SkillProfile,
)
from dds_ai.contracts.remarks import Remark, RemarkType, Severity
from dds_ai.contracts.scoring import ScoreSummary, ScoreVersion, Verdict

from .conftest import T0, ev

AI = Path(__file__).resolve().parents[1]
LLM = ModelRef(component="llm", model_name="m", model_version="1", prompt_version="supervisor-1")
RULES = VersionRef(name="rules", version="1")


def test_reply_cannot_use_unpublished_facts_or_actions():
    who = Interlocutor(
        role_id="supervisor",
        published_fact_ids=("address",),
        allowed_actions=(SupervisorAction.CLARIFY,),
    )
    ok = SupervisorReply(
        action=SupervisorAction.CLARIFY,
        text="Уточните адрес",
        used_fact_ids=("address",),
        mode=ReplyMode.LLM,
        model_ref=LLM,
    )
    ok.check_against(who)
    with pytest.raises(ValueError, match="unpublished"):
        SupervisorReply(
            action=SupervisorAction.CLARIFY,
            text="Пострадавших трое?",
            used_fact_ids=("victims",),
            mode=ReplyMode.FALLBACK,
        ).check_against(who)
    with pytest.raises(ValueError, match="not allowed"):
        SupervisorReply(
            action=SupervisorAction.CONFIRM_RECEIPT, text="Принято", mode=ReplyMode.FALLBACK
        ).check_against(who)


def test_llm_reply_requires_model_version():
    with pytest.raises(ValidationError):
        SupervisorReply(action=SupervisorAction.LISTEN, text="Слушаю", mode=ReplyMode.LLM)


def test_new_trainee_has_no_invented_level():
    SkillProfile(
        profile_version_id=uuid4(),
        trainee_id=uuid4(),
        dds_profile="fire",
        status=ProfileStatus.INSUFFICIENT_DATA,
        aggregation_rules=RULES,
        created_at=T0,
    )
    with pytest.raises(ValidationError):
        SkillProfile(
            profile_version_id=uuid4(),
            trainee_id=uuid4(),
            dds_profile="fire",
            status=ProfileStatus.OK,
            groups=(GroupLevel(group_id="card", level=0.8, attempts_used=1),),
            aggregation_rules=RULES,
            created_at=T0,
        )


def rec(**kw):
    base = {
        "recommendation_id": uuid4(),
        "trainee_id": uuid4(),
        "profile_version_id": None,
        "pool_snapshot_id": uuid4(),
        "pool_task_ids": ("t1", "t2"),
        "reason": "r",
        "selection_rules": RULES,
        "created_at": T0,
    }
    return Recommendation(**(base | kw))


def test_recommendation_stays_within_pool():
    rec(status=RecommendationStatus.AVAILABLE, task_id="t2")
    with pytest.raises(ValidationError, match="pool"):
        rec(status=RecommendationStatus.AVAILABLE, task_id="generated-on-the-fly")
    with pytest.raises(ValidationError):
        rec(status=RecommendationStatus.UNAVAILABLE, task_id="t1")


def test_score_correction_needs_reason_and_summary_shape():
    summary = ScoreSummary(rubric_version="w01", verdict=Verdict.PASSED, total=90)
    common = {
        "score_version_id": uuid4(),
        "attempt_id": uuid4(),
        "author": "teacher",
        "created_at": T0,
        "criterion_results": (),
        "summary": summary,
    }
    ScoreVersion(**common)
    with pytest.raises(ValidationError, match="reason"):
        ScoreVersion(**common, parent_score_version_id=uuid4())
    with pytest.raises(ValidationError):
        ScoreSummary(rubric_version="w01", verdict=Verdict.PROVISIONAL, total=80)
    with pytest.raises(ValidationError):
        ScoreSummary(rubric_version="w01", verdict=Verdict.NOT_SCORED, total=0)


@pytest.mark.parametrize(
    ("rtype", "severity", "counts"),
    [
        (RemarkType.FACT_FIELD, Severity.ERROR, True),
        (RemarkType.GRAMMAR, Severity.INFO, False),
        (RemarkType.TECHNICAL_FAULT, Severity.MAJOR_ERROR, False),
        (RemarkType.SOURCE_UNCERTAINTY, Severity.ERROR, False),
    ],
)
def test_technical_fault_is_not_student_error(rtype, severity, counts):
    r = Remark(
        remark_id="r",
        attempt_id="a",
        type=rtype,
        severity=severity,
        text="t",
        evidence=ev(),
        owner="rule",
    )
    assert r.counts_as_error is counts


def test_transcript_evidence_requires_interval():
    with pytest.raises(ValidationError):
        Evidence(kind=EvidenceKind.TRANSCRIPT_SPAN, ref="u1")
    Evidence(
        kind=EvidenceKind.TRANSCRIPT_SPAN, ref="u1", start_ms=100, end_ms=900, excerpt="горит кухня"
    )


def test_json_schemas_are_up_to_date():
    proc = subprocess.run(
        [sys.executable, str(AI / "tools" / "export_schemas.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
