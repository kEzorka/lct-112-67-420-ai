"""Проверка выхода LLM-формулировки реплики руководителя (M3, 6.4, инварианты 1–3, 5).

Выход модели — JSON `{"action": ..., "text": ...}`. Реплика принимается, только если:
- схема верна, действие совпадает с выбранным автоматом, длина в пределах;
- в тексте нет чисел (цифрами и словами), адресных элементов, имён собственных, служб и
  латиницы, которых нет в разрешённом контексте (черновик, использованные факты сценария,
  сведения, переданные диспетчером в этом вызове);
- нет разговора об оценке, эталоне, сценарии, «правильном» или «ожидаемом» решении,
  инструкциях и других служебных темах, если их нет в черновике (реплика диспетчера такие
  слова в разрешённый контекст не добавляет);
- смысл действия сохранён: «неизвестно» не превращается в число, числа черновика не
  теряются, получение доклада подтверждается только действием `confirm_receipt`.

Проверки консервативны: отклонённая реплика заменяется проверенным шаблоном (C-04), поэтому
ложное отклонение стоит только живости формулировки, а пропуск — выдуманного факта.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum

from .. import facts_guard
from ..contracts.dialogue import SupervisorAction
from . import matching


class RejectReason(StrEnum):
    EMPTY = "empty"
    NOT_JSON = "not_json"
    SCHEMA = "schema"
    ACTION_MISMATCH = "action_mismatch"
    TOO_LONG = "too_long"
    MARKUP = "markup"
    NUMBER_OUTSIDE_CONTEXT = "number_outside_context"
    NUMBER_WORD_OUTSIDE_CONTEXT = "number_word_outside_context"
    ADDRESS_OUTSIDE_CONTEXT = "address_outside_context"
    NAME_OUTSIDE_CONTEXT = "name_outside_context"
    SERVICE_OUTSIDE_CONTEXT = "service_outside_context"
    FOREIGN_TEXT = "foreign_text"
    META_TALK = "meta_talk"
    UNKNOWN_DROPPED = "unknown_dropped"
    NUMBER_DROPPED = "number_dropped"
    ACCEPTANCE_OUTSIDE_CONFIRM = "acceptance_outside_confirm"
    ACCEPTANCE_MISSING = "acceptance_missing"
    NOT_A_QUESTION = "not_a_question"


class InvalidReply(ValueError):
    def __init__(self, reason: RejectReason, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else str(reason))
        self.reason = reason


# --- словари ------------------------------------------------------------------------------
# Числа/адреса/службы/имена вне контекста — общая проверка facts_guard (E-5); здесь остаются
# только реплико-специфичные словари: служебные темы, согласие, уточнение.

_META_STEMS = (
    "эталон",
    "оценк",
    "оцени",
    "балл",
    "промпт",
    "инструкц",
    "критери",
    "систем",
    "зачет",
    "зачт",
    "правильн",
    "игнорир",
    "модел",
    "настройк",
    "нейросет",
    "сценари",
    "ожидаем",
    "решени",  # суждение о решении ДДС — только если оно есть в черновике
)
_ACCEPT_STEMS = ("принят", "принял", "принима", "получил", "получен", "подтвержда")
_CONFIRM_STEMS = ("принят", "принял", "получил", "получен")
_UNKNOWN_STEMS = ("неизвестн", "не известн")
_CLARIFY_STEMS = ("уточн", "поясн", "повтор", "назов", "сообщ", "скаж", "поправ", "что", "как")
_ALWAYS_ALLOWED = "ддс"

_MARKUP_RE = re.compile(r"[<>{}]")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")

_FACT_REASON = {
    facts_guard.FactViolationKind.NUMBER: RejectReason.NUMBER_OUTSIDE_CONTEXT,
    facts_guard.FactViolationKind.NUMBER_WORD: RejectReason.NUMBER_WORD_OUTSIDE_CONTEXT,
    facts_guard.FactViolationKind.ADDRESS: RejectReason.ADDRESS_OUTSIDE_CONTEXT,
    facts_guard.FactViolationKind.SERVICE: RejectReason.SERVICE_OUTSIDE_CONTEXT,
    facts_guard.FactViolationKind.FOREIGN_TEXT: RejectReason.FOREIGN_TEXT,
    facts_guard.FactViolationKind.NAME: RejectReason.NAME_OUTSIDE_CONTEXT,
}


def _has_stem(toks: list[str] | set[str], stems: tuple[str, ...]) -> bool:
    return any(t.startswith(stems) for t in toks)


def _stems_in(toks: list[str], stems: tuple[str, ...]) -> set[str]:
    return {s for s in stems for t in toks if t.startswith(s)}


def parse_output(raw: str) -> tuple[str, str]:
    """JSON-объект ровно с ключами action и text. Допускается обёртка ```json ... ```."""
    body = _FENCE.sub("", raw.strip()).strip()
    if not body:
        raise InvalidReply(RejectReason.EMPTY)
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise InvalidReply(RejectReason.NOT_JSON) from None
    if not isinstance(data, dict) or set(data) != {"action", "text"}:
        raise InvalidReply(RejectReason.SCHEMA, "expected keys action, text")
    action, text = data["action"], data["text"]
    if not isinstance(action, str) or not isinstance(text, str):
        raise InvalidReply(RejectReason.SCHEMA, "action and text must be strings")
    return action, text


def check_text(
    text: str,
    *,
    action: SupervisorAction,
    draft: str,
    context: str,
    max_chars: int,
) -> str:
    """Проверить формулировку реплики против черновика и разрешённого контекста."""
    text = text.strip()
    if not text:
        raise InvalidReply(RejectReason.EMPTY)
    if len(text) > max_chars:
        raise InvalidReply(RejectReason.TOO_LONG, f"> {max_chars}")
    if _MARKUP_RE.search(text):
        raise InvalidReply(RejectReason.MARKUP)

    allowed = f"{draft} {context} {_ALWAYS_ALLOWED}"
    draft_toks = matching.tokens(draft)
    toks = matching.tokens(text)

    violation = facts_guard.find_violation(text, allowed=allowed)
    if violation is not None:
        raise InvalidReply(_FACT_REASON[violation.kind], violation.detail)
    # Служебные темы — только если они есть в черновике: реплика диспетчера их не разрешает.
    if _stems_in(toks, _META_STEMS) - _stems_in(draft_toks, _META_STEMS):
        raise InvalidReply(RejectReason.META_TALK)

    norm_text, norm_draft = matching.normalize(text), matching.normalize(draft)
    if any(s in norm_draft for s in _UNKNOWN_STEMS) and not any(
        s in norm_text for s in _UNKNOWN_STEMS
    ):
        raise InvalidReply(RejectReason.UNKNOWN_DROPPED)
    if matching.digits(draft) - matching.digits(text):
        raise InvalidReply(RejectReason.NUMBER_DROPPED)

    if action is SupervisorAction.CONFIRM_RECEIPT:
        if not _has_stem(toks, _CONFIRM_STEMS):
            raise InvalidReply(RejectReason.ACCEPTANCE_MISSING)
    elif _stems_in(toks, _ACCEPT_STEMS) - _stems_in(draft_toks, _ACCEPT_STEMS):
        raise InvalidReply(RejectReason.ACCEPTANCE_OUTSIDE_CONFIRM)
    if action is SupervisorAction.READ_BACK and "?" not in text:
        raise InvalidReply(RejectReason.NOT_A_QUESTION)
    if (
        action is SupervisorAction.CLARIFY
        and "?" not in text
        and not _has_stem(toks, _CLARIFY_STEMS)
    ):
        raise InvalidReply(RejectReason.NOT_A_QUESTION)
    return text


def validate_output(
    raw: str,
    *,
    action: SupervisorAction,
    draft: str,
    context: str,
    max_chars: int,
) -> str:
    """Разобрать и проверить выход модели; вернуть текст реплики или InvalidReply."""
    claimed, text = parse_output(raw)
    if claimed != action.value:
        raise InvalidReply(RejectReason.ACTION_MISMATCH, f"{claimed!r} != {action.value!r}")
    return check_text(text, action=action, draft=draft, context=context, max_chars=max_chars)
