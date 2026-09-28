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

_NUMBER_WORDS = frozenset(
    """
    ноль нуля один одна одно одного одной одному одним одну два две двое двух двум двумя двоих
    три трое трех трем тремя троих четыре четверо четырех четырем четырьмя четверых пять
    пятеро пяти пятью пятерых шесть шестеро шести шестью семь семеро семи семью восемь
    восьми восемью девять девяти девятью десять десяти десятью сто ста двести триста
    четыреста пятьсот шестьсот семьсот восемьсот девятьсот тысяча тысячи тысяч тысячу сорок
    сорока девяносто девяноста полтора полторы несколько нескольких десяток десятка десятки
    дюжина сотня сотни сотен первый второй третий четвертый пятый шестой седьмой восьмой
    девятый десятый
    """.split()
)
_NUMBER_WORD_RE = re.compile(r"^\w+(дцат|десят)\w*$")

# Адресные элементы по группам: «доме» и «дома» — один элемент, «проспект» и «улица» — разные.
_ADDRESS_GROUPS = {
    "street": "ул улица улице улицу улицы",
    "avenue": "проспект проспекте просп пр",
    "lane": "переулок переулке пер",
    "highway": "шоссе",
    "boulevard": "бульвар бульваре",
    "square": "площадь площади",
    "house": "дом доме дома д",
    "building": "корпус корп строение стр",
    "flat": "кв квартира квартире квартиры квартиру",
    "district": "мкр микрорайон микрорайоне",
    "city": "г город города городе",
    "settlement": "поселок поселке пос деревня деревне село селе",
    "entrance": "подъезд подъезде",
    "floor": "этаж этаже этажа этажей",
}
_ADDRESS = {w: group for group, words in _ADDRESS_GROUPS.items() for w in words.split()}
_SERVICE_STEMS = (
    "пожарн",
    "полиц",
    "газов",
    "газовщ",
    "мчс",
    "спасат",
    "росгвард",
    "медик",
    "медицин",
    "реанимац",
    "скорая",
    "скорой",
    "скорую",
    "энергосбыт",
    "водоканал",
    "эвакуатор",
)
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
_ALWAYS_ALLOWED = frozenset({"ддс"})

_CAPITALIZED = re.compile(r"[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z-]*")
_LATIN = re.compile(r"[A-Za-z]")
_MARKUP_RE = re.compile(r"[<>{}]")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _has_stem(toks: list[str] | set[str], stems: tuple[str, ...]) -> bool:
    return any(t.startswith(stems) for t in toks)


def _stems_in(toks: list[str], stems: tuple[str, ...]) -> set[str]:
    return {s for s in stems for t in toks if t.startswith(s)}


def _is_number_word(tok: str) -> bool:
    return tok in _NUMBER_WORDS or bool(_NUMBER_WORD_RE.match(tok))


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip()
    return not before or before[-1] in ".!?"


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

    allowed = f"{draft} {context}"
    allowed_toks = set(matching.tokens(allowed)) | _ALWAYS_ALLOWED
    draft_toks = matching.tokens(draft)
    toks = matching.tokens(text)
    new = [t for t in toks if t not in allowed_toks]

    extra_digits = matching.digits(text) - matching.digits(allowed)
    if extra_digits:
        raise InvalidReply(RejectReason.NUMBER_OUTSIDE_CONTEXT, ",".join(sorted(extra_digits)))
    if any(_is_number_word(t) for t in new):
        raise InvalidReply(RejectReason.NUMBER_WORD_OUTSIDE_CONTEXT)
    allowed_address = {_ADDRESS[t] for t in allowed_toks if t in _ADDRESS}
    if {_ADDRESS[t] for t in toks if t in _ADDRESS} - allowed_address:
        raise InvalidReply(RejectReason.ADDRESS_OUTSIDE_CONTEXT)
    if _stems_in(toks, _SERVICE_STEMS) - _stems_in(matching.tokens(allowed), _SERVICE_STEMS):
        raise InvalidReply(RejectReason.SERVICE_OUTSIDE_CONTEXT)
    if _LATIN.search(text) and not set(_LATIN.findall(text)) <= set(_LATIN.findall(allowed)):
        raise InvalidReply(RejectReason.FOREIGN_TEXT)
    for m in _CAPITALIZED.finditer(text):
        word = matching.normalize(m.group())
        if not _sentence_start(text, m.start()) and not all(
            t in allowed_toks for t in matching.tokens(word)
        ):
            raise InvalidReply(RejectReason.NAME_OUTSIDE_CONTEXT)
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
