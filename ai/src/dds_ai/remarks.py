"""Сборка замечаний попытки (D-037): грамматика ручного ввода, технические сбои,
неопределённость источника. `technical_fault` и `source_uncertainty` — не ошибки ученика
(`Remark.counts_as_error`), считаются отдельно от группового анализа ошибок (6.9).
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from .contracts.card import CardField, FieldOrigin, FieldState
from .contracts.common import Evidence, EvidenceKind, VersionRef
from .contracts.criteria import CriterionResult, CriterionStatus, PartialReason
from .contracts.events import AttemptEvent, ModelFailure
from .contracts.remarks import Remark, RemarkType, Severity
from .contracts.routing import RoutingDecision
from .grammar.checker import GrammarIssue, check_text

GRAMMAR_RULES = VersionRef(name="manual-grammar-rules", version="1")
GRAMMAR_CRITERION = "manual.grammar"


def _grammar_remark(
    *, remark_id: str, attempt_id: UUID, field_name: str, issue: GrammarIssue
) -> Remark:
    return Remark(
        remark_id=remark_id,
        attempt_id=str(attempt_id),
        type=RemarkType.GRAMMAR,
        severity=Severity.ERROR if issue.counts_against_score else Severity.INFO,
        criterion_id=GRAMMAR_CRITERION,
        text=f"{field_name}: {issue.kind.value} — «{issue.fragment}»",
        suggestion=issue.suggestion,
        evidence=(Evidence(kind=EvidenceKind.CARD_FIELD, ref=field_name, excerpt=issue.fragment),),
        owner="rule",
    )


def dispatcher_text_fields(card_fields: dict[str, CardField]) -> dict[str, str]:
    """Свободнотекстовые поля, которые фактически заполнил/изменил диспетчер (D-034):
    только они проверяются на грамотность и семантику ручных дополнений — не дефект
    исходной карточки от ИИ-оператора 112."""
    return {
        name: f.raw
        for name, f in card_fields.items()
        if f.origin is FieldOrigin.DISPATCHER
        and f.state is FieldState.KNOWN
        and isinstance(f.raw, str)
    }


def grammar_remarks(card_fields: dict[str, CardField], *, attempt_id: UUID) -> tuple[Remark, ...]:
    out: list[Remark] = []
    for field_name, text in sorted(dispatcher_text_fields(card_fields).items()):
        for i, issue in enumerate(check_text(text)):
            out.append(
                _grammar_remark(
                    remark_id=f"grammar-{field_name}-{i}",
                    attempt_id=attempt_id,
                    field_name=field_name,
                    issue=issue,
                )
            )
    return tuple(out)


def grammar_criterion(
    card_fields: dict[str, CardField], remarks: Sequence[Remark]
) -> CriterionResult:
    """manual.grammar по замечаниям локальной проверки — правило, не модель. Порог PASSED/0,5/
    FAILED — рабочая конвенция команды (грамматика не входит в пороги W-01)."""
    fields = dispatcher_text_fields(card_fields)
    if not fields:
        return CriterionResult(
            criterion_id=GRAMMAR_CRITERION,
            status=CriterionStatus.NOT_CHECKED,
            explanation="Нет ручного текстового ввода диспетчера — проверять грамотность нечего.",
        )
    scoring = [r for r in remarks if r.criterion_id == GRAMMAR_CRITERION and r.counts_as_error]
    evidence = tuple(r.evidence[0] for r in remarks if r.criterion_id == GRAMMAR_CRITERION) or (
        Evidence(kind=EvidenceKind.CARD_FIELD, ref=sorted(fields)[0]),
    )
    if not scoring:
        return CriterionResult(
            criterion_id=GRAMMAR_CRITERION,
            status=CriterionStatus.PASSED,
            value=1,
            evidence=evidence,
            rule_ref=GRAMMAR_RULES,
            explanation="Механических ошибок в ручном вводе не найдено.",
        )
    if len(scoring) == 1:
        return CriterionResult(
            criterion_id=GRAMMAR_CRITERION,
            status=CriterionStatus.PASSED,
            value=0.5,
            partial_reason=PartialReason.INACCURATE_FIELD,
            evidence=evidence,
            rule_ref=GRAMMAR_RULES,
            explanation=f"Одна ошибка в ручном вводе: {scoring[0].text}.",
        )
    return CriterionResult(
        criterion_id=GRAMMAR_CRITERION,
        status=CriterionStatus.FAILED,
        value=0,
        evidence=evidence,
        rule_ref=GRAMMAR_RULES,
        explanation=f"{len(scoring)} ошибки/ошибок в ручном вводе.",
    )


def technical_fault_remarks(
    events: Sequence[AttemptEvent], *, attempt_id: UUID
) -> tuple[Remark, ...]:
    """Сбой модели — техническая неисправность, не ошибка ученика (инвариант 4)."""
    out = []
    for e in events:
        if not isinstance(e, ModelFailure):
            continue
        text = f"{e.component.value}: {e.kind.value}"
        if e.recovered:
            text += " (восстановлено повтором)"
        out.append(
            Remark(
                remark_id=f"technical-{e.event_id}",
                attempt_id=str(attempt_id),
                type=RemarkType.TECHNICAL_FAULT,
                severity=Severity.INFO if e.recovered else Severity.ERROR,
                text=text,
                evidence=(Evidence(kind=EvidenceKind.EVENT, ref=str(e.event_id)),),
                owner="rule",
            )
        )
    return tuple(out)


def source_uncertainty_remarks(routing: RoutingDecision, *, attempt_id: UUID) -> tuple[Remark, ...]:
    """Неопределённость источника (ambiguous/quarantined правило) — не ошибка ученика."""
    if routing.auto_gradable:
        return ()
    ref = f"{routing.routing_rules.version}:{routing.rule_id or '—'}"
    return (
        Remark(
            remark_id=f"source-uncertainty-{routing.rule_id or 'none'}",
            attempt_id=str(attempt_id),
            type=RemarkType.SOURCE_UNCERTAINTY,
            severity=Severity.INFO,
            criterion_id="routing.services",
            text=f"Правило маршрутизации {ref} не active ({routing.rule_status}) — решает эксперт.",
            evidence=(Evidence(kind=EvidenceKind.RULE, ref=ref),),
            owner="expert",
        ),
    )


def build_remarks(
    *,
    card_fields: dict[str, CardField],
    events: Sequence[AttemptEvent],
    routing: RoutingDecision,
    attempt_id: UUID,
) -> tuple[Remark, ...]:
    return (
        grammar_remarks(card_fields, attempt_id=attempt_id)
        + technical_fault_remarks(events, attempt_id=attempt_id)
        + source_uncertainty_remarks(routing, attempt_id=attempt_id)
    )
