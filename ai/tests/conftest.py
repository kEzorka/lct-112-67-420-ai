from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dds_ai.contracts.common import Evidence, EvidenceKind, VersionRef
from dds_ai.contracts.criteria import CriterionResult, CriterionStatus, PartialReason
from dds_ai.contracts.rubric import Rubric

CONFIG = Path(__file__).resolve().parents[1] / "config"
T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
RULE = VersionRef(name="rubric-rule", version="1")


@pytest.fixture
def rubric() -> Rubric:
    return Rubric.model_validate(json.loads((CONFIG / "rubric.w01.json").read_text("utf-8")))


def ev(ref: str = "e1") -> tuple[Evidence, ...]:
    return (Evidence(kind=EvidenceKind.EVENT, ref=ref),)


def result(cid: str, status: CriterionStatus, value=None, partial: PartialReason | None = None):
    verified = status in (CriterionStatus.PASSED, CriterionStatus.FAILED, CriterionStatus.NOT_DONE)
    if verified and value is None:
        value = 1 if status is CriterionStatus.PASSED else 0
    return CriterionResult(
        criterion_id=cid,
        status=status,
        value=value,
        partial_reason=partial,
        evidence=ev() if verified else (),
        rule_ref=RULE if verified else None,
    )


def all_results(rubric: Rubric, status: CriterionStatus, **overrides: CriterionResult):
    out = []
    for g in rubric.groups:
        for c in g.criteria:
            out.append(overrides.get(c.criterion_id) or result(c.criterion_id, status))
    return out
