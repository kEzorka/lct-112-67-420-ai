"""Частота типичных ошибок группы (6.9, [ТЗ]): по типам D-037 и группам рубрики, по профилю
ДДС и сложности. Считается только по `valid_score` (инвариант 13, C-07): попытка,
технически нарушенная или ушедшая в `provisional`, не в состоянии стать ошибкой ученика —
это находка калибровки M5, а не ошибка. Технические сбои и доля `not_checked` показываются
отдельно и по всем рассмотренным попыткам, чтобы не спрятать их внутри счётчика ошибок.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from ..contracts.analytics import ErrorFrequencyBucket, GroupErrorReport
from ..contracts.criteria import CriterionResult, CriterionStatus
from ..contracts.remarks import Remark, RemarkType
from ..contracts.rubric import Rubric


@dataclass(frozen=True)
class AttemptErrorRecord:
    """Одна рассмотренная попытка: сегмент, валидность оценки, замечания и результаты критериев."""

    attempt_id: UUID
    dds_profile: str
    difficulty_band: str
    valid_score: bool
    remarks: tuple[Remark, ...] = ()
    criterion_results: tuple[CriterionResult, ...] = ()


def _group_of(rubric: Rubric, criterion_id: str | None) -> str | None:
    if criterion_id is None:
        return None
    try:
        return rubric.group_of(criterion_id).group_id
    except KeyError:
        return None


def build_group_error_report(
    records: Sequence[AttemptErrorRecord], rubric: Rubric, *, now: datetime
) -> GroupErrorReport:
    counts: dict[tuple[RemarkType, str | None, str, str], int] = defaultdict(int)
    attempt_ids: dict[tuple[RemarkType, str | None, str, str], set[UUID]] = defaultdict(set)
    technical_fault_count = 0
    not_checked_count = 0
    total_criteria = 0
    valid_attempts = 0

    for rec in records:
        if rec.valid_score:
            valid_attempts += 1
            for r in rec.remarks:
                if not r.counts_as_error:
                    continue
                key = (
                    r.type,
                    _group_of(rubric, r.criterion_id),
                    rec.dds_profile,
                    rec.difficulty_band,
                )
                counts[key] += 1
                attempt_ids[key].add(rec.attempt_id)
        # технические сбои и «не проверено» — по всем рассмотренным попыткам, не только valid_score
        technical_fault_count += sum(1 for r in rec.remarks if r.type is RemarkType.TECHNICAL_FAULT)
        total_criteria += len(rec.criterion_results)
        not_checked_count += sum(
            1 for c in rec.criterion_results if c.status is CriterionStatus.NOT_CHECKED
        )

    buckets = tuple(
        ErrorFrequencyBucket(
            remark_type=remark_type,
            group_id=group_id,
            dds_profile=dds_profile,
            difficulty_band=difficulty_band,
            count=count,
            attempt_ids=tuple(
                sorted(attempt_ids[(remark_type, group_id, dds_profile, difficulty_band)])
            ),
        )
        for (remark_type, group_id, dds_profile, difficulty_band), count in sorted(
            counts.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][0].value, kv[0][1] or "")
        )
    )
    return GroupErrorReport(
        generated_at=now,
        rubric_version=rubric.rubric_version,
        buckets=buckets,
        technical_fault_count=technical_fault_count,
        not_checked_count=not_checked_count,
        total_criteria_count=total_criteria,
        total_valid_attempts=valid_attempts,
    )
