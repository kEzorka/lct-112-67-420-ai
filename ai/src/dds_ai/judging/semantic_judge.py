"""Семантический оцениватель (6.6, D-031, D-034, D-036) через `LLMProvider`.

Реализует `ports.SemanticJudge`: вызывается изнутри `evaluation.judge_criterion`, которая уже
защищена `FaultInjector`/`InferenceWorker` (тайм-аут, очередь, полоса `background`) —
`ValidationError`/`InvalidOutput` отсюда превращаются в `not_checked`, не в 0 (инвариант 4).

Контекст задаёт вызывающий (`cycle/rules.py`) через ключ `context["kind"]`:
- `"field_fact"` — структурное/свободнотекстовое поле карточки против фактов эталона (D-034):
  `field_name`, `student_text` (`None`/пусто → поле недоступно, не штраф), `expected_facts`
  (`[{"label": ..., "values": [...]}]`, `values` уже включает разрешённые вариации).
- `"conversation_coverage"` — элементы доклада руководителю против транскрипта (D-036):
  `transcript` (`[{"event_id": ..., "text": ...}]`, только реплики диспетчера), `items`
  (`[{"item_id": ..., "description": ...}]`).

Модель видит эталон осознанно (в отличие от промпта руководителя, инвариант 1 которого сюда
не относится: оцениватель не разговаривает с обучаемым и не пишет в карточку/реплики, только
в `CriterionResult`). Каждая цитата в выходе модели обязана быть дословной подстрокой
проверяемого текста — иначе выход недостоверен (`InvalidOutput`): оцениватель не должен
изобретать доказательство, которого нет у ученика.

Устойчивость к перефразированию и повреждённому транскрипту: `conversation_coverage`
никогда не выставляет `failed` — только `passed` (все элементы подтверждены семантически)
или `not_checked` (что-то не подтверждено — паравраза не распознана, транскрипт неразборчив
или повреждён), поэтому STT-устойчивость (раздел 8 промпта) не может стать штрафом только
из-за того, что ключевые слова M1 не совпали.

`field_fact` содержательно может отличить неверный ответ (сравнение со структурированным/явным
фактом эталона), но пока согласие оценивателя с рабочей разметкой ниже порога W-03 (решение
F-2, `docs/ai/progress.md`, волна 3; калибровка — `docs/ai/calibration-report.md`), модель
самостоятельно ставит `failed` только тем критериям, что явно перечислены в `JudgePolicy`
(`policy.py`, конфиг `ai/config/judge.policy.json`) — по умолчанию список пуст. Для остальных
уверенное `none` уходит в `not_checked` с объяснением «ждёт эксперта», как уже было устроено
для `voice.facts_transferred`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from pydantic import ValidationError

from ..contracts.common import Evidence, EvidenceKind, ModelRef
from ..contracts.criteria import CriterionResult, CriterionStatus, PartialReason
from ..faults import InvalidOutput
from ..ports import LLMProvider
from .policy import JudgePolicy
from .prompts import JudgePromptTemplate
from .schema import REASON_CODES, ConversationVerdict, FieldVerdict, MatchLevel

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_FIRST_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(raw: str) -> dict:
    """Снять возможную обёртку ```json ... ``` (модели на CPU часто её добавляют вопреки
    инструкции — наблюдение бенчмарка, `docs/ai/bench/llm-cpu.md`) и разобрать JSON-объект."""
    body = _FENCE.sub("", raw.strip()).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        m = _FIRST_OBJECT.search(body)
        if m is None:
            raise InvalidOutput("judge output is not a JSON object") from None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError as exc:
            raise InvalidOutput("judge output is not a JSON object") from exc
    if not isinstance(data, dict):
        raise InvalidOutput("judge output is not a JSON object")
    return data


def _quote_ok(quote: str, source: str) -> bool:
    return not quote.strip() or quote.strip().lower() in source.lower()


def _find_source_event(quote: str, transcript: Sequence[dict]) -> str | None:
    q = quote.strip().lower()
    if not q:
        return None
    for entry in transcript:
        if q in str(entry.get("text", "")).lower():
            return str(entry.get("event_id"))
    return None


class LLMSemanticJudge:
    """Судит по одному критерию за раз (D-031: поштучно, с доказательством)."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        prompt: JudgePromptTemplate | None = None,
        policy: JudgePolicy | None = None,
    ):
        self.llm = llm
        self.prompt = prompt or JudgePromptTemplate.load()
        self.policy = policy or JudgePolicy.load()
        self.model_ref = ModelRef(
            component="semantic_judge",
            model_name=llm.model_ref.model_name,
            model_version=llm.model_ref.model_version,
            prompt_version=self.prompt.version,
        )

    def judge(self, criterion_id: str, context: dict) -> CriterionResult:
        kind = context.get("kind")
        if kind == "field_fact":
            return self._judge_field(criterion_id, context)
        if kind == "conversation_coverage":
            return self._judge_conversation(criterion_id, context)
        raise InvalidOutput(f"semantic judge has no handler for context kind {kind!r}")

    # --- D-034: поле карточки против фактов эталона ---------------------------------------

    def _judge_field(self, criterion_id: str, context: dict) -> CriterionResult:
        field_name = str(context.get("field_name") or criterion_id)
        student_text = context.get("student_text")
        expected_facts = context.get("expected_facts") or []
        if not student_text or not str(student_text).strip():
            return CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.NOT_CHECKED,
                explanation=f"Поле «{field_name}» недоступно диспетчеру или не заполнено.",
            )
        if not expected_facts:
            return CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.NOT_CHECKED,
                explanation="Эталон не определяет ожидаемые факты для этого поля.",
            )
        student_text = str(student_text)
        expected_text = "; ".join(
            f"{f['label']}: {' / '.join(f['values'])}" for f in expected_facts
        )
        raw = self.llm.complete(
            self.prompt.field_prompt(expected=expected_text, student_text=student_text),
            max_tokens=self.prompt.max_tokens_field,
        )
        try:
            verdict = FieldVerdict.model_validate(_parse_json_object(raw))
        except ValidationError as exc:
            raise InvalidOutput(f"judge field verdict does not match schema: {exc}") from exc
        if verdict.reason_code not in REASON_CODES:
            raise InvalidOutput(f"unknown reason_code {verdict.reason_code!r}")
        if not _quote_ok(verdict.quote, student_text):
            raise InvalidOutput("judge quote is not a verbatim substring of the checked text")

        evidence = (
            Evidence(
                kind=EvidenceKind.CARD_FIELD,
                ref=field_name,
                excerpt=verdict.quote or student_text[:200],
            ),
        )
        explanation = f"«{field_name}»: {verdict.reason_code} (семантическая проверка, M5)."
        if verdict.match is MatchLevel.FULL:
            return CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.PASSED,
                value=1,
                evidence=evidence,
                model_ref=self.model_ref,
                decided_by="model",
                explanation=explanation,
            )
        if verdict.match is MatchLevel.PARTIAL:
            return CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.PASSED,
                value=0.5,
                partial_reason=PartialReason.INACCURATE_FIELD,
                evidence=evidence,
                model_ref=self.model_ref,
                decided_by="model",
                explanation=explanation,
            )
        if verdict.match is MatchLevel.NONE:
            if criterion_id in self.policy.failable_criteria:
                return CriterionResult(
                    criterion_id=criterion_id,
                    status=CriterionStatus.FAILED,
                    value=0,
                    evidence=evidence,
                    model_ref=self.model_ref,
                    decided_by="model",
                    explanation=explanation,
                )
            # F-2: пока согласие ниже W-03, некалиброванный критерий не проваливает модель
            # самостоятельно — уверенное несовпадение ждёт эксперта, не штраф (инвариант 4).
            return CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.NOT_CHECKED,
                explanation=(
                    f"«{field_name}»: модель уверенно не подтверждает соответствие эталону "
                    f"({verdict.reason_code}), но самостоятельный «не выполнено» по этому "
                    "критерию пока не разрешён (F-2, согласие оценивателя ниже порога W-03) — "
                    "ждёт эксперта."
                ),
            )
        return CriterionResult(
            criterion_id=criterion_id,
            status=CriterionStatus.NOT_CHECKED,
            explanation=f"«{field_name}»: текст неразборчив или повреждён для сравнения (M5).",
        )

    # --- D-036: доклад руководителю против транскрипта -------------------------------------

    def _judge_conversation(self, criterion_id: str, context: dict) -> CriterionResult:
        transcript = list(context.get("transcript") or [])
        items = list(context.get("items") or [])
        if not items:
            return CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.NOT_CHECKED,
                explanation="Нет проверяемых элементов доклада для семантической проверки.",
            )
        transcript_text = " ".join(str(e.get("text", "")) for e in transcript)
        if not transcript_text.strip():
            return CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.NOT_CHECKED,
                explanation="Транскрипт пуст — семантическая проверка недоступна.",
            )
        items_text = "\n".join(f"{i['item_id']}: {i['description']}" for i in items)
        raw = self.llm.complete(
            self.prompt.conversation_prompt(items=items_text, transcript=transcript_text),
            max_tokens=self.prompt.max_tokens_conversation,
        )
        try:
            verdict = ConversationVerdict.model_validate(_parse_json_object(raw))
        except ValidationError as exc:
            raise InvalidOutput(f"judge conversation verdict does not match schema: {exc}") from exc

        by_id = {v.item_id: v for v in verdict.items}
        expected_ids = {i["item_id"] for i in items}
        if set(by_id) != expected_ids:
            raise InvalidOutput("judge answered a different set of item_id than requested")
        for v in verdict.items:
            if not _quote_ok(v.quote, transcript_text):
                raise InvalidOutput("judge quote is not a verbatim substring of the transcript")

        unconfirmed = (MatchLevel.NONE, MatchLevel.UNCLEAR)
        unresolved = sorted(i["item_id"] for i in items if by_id[i["item_id"]].match in unconfirmed)
        if unresolved:
            # Ни разу не failed: паравраза/повреждённый транскрипт не штраф (инвариант 4).
            return CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.NOT_CHECKED,
                explanation=(
                    "Семантическая проверка не подтверждает элементы доклада: "
                    + ", ".join(unresolved)
                    + ". Ожидает эксперта."
                ),
            )
        evidence = tuple(
            Evidence(
                kind=EvidenceKind.EVENT,
                ref=_find_source_event(by_id[i["item_id"]].quote, transcript) or "transcript",
                excerpt=by_id[i["item_id"]].quote or None,
            )
            for i in items
        )
        return CriterionResult(
            criterion_id=criterion_id,
            status=CriterionStatus.PASSED,
            value=1,
            evidence=evidence,
            model_ref=self.model_ref,
            decided_by="model",
            explanation=(
                "Существенные сведения переданы (семантическая проверка, перефразирование учтено)."
            ),
        )
