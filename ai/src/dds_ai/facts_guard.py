"""Общая проверка «нет фактов вне контекста» для выхода LLM (решение E-5, инвариант 5).

Числа (цифрами и словами), адресные элементы, названия служб, имена собственные и латиница,
которых нет в разрешённом контексте, — выдуманные факты. Раньше проверку делал только
`supervisor/validator.py` (реплика руководителя); теперь её же используют переформулировка
описания карточки (`scenarios/card_generation.py`) и семантический оцениватель (`judging/`),
чтобы поведение не расходилось между модулями.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from . import text_norm as matching

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
ADDRESS_GROUPS: dict[str, str] = {
    w: g for g, words in _ADDRESS_GROUPS.items() for w in words.split()
}

SERVICE_STEMS = (
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

_CAPITALIZED = re.compile(r"[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z-]*")
_LATIN = re.compile(r"[A-Za-z]")


class FactViolationKind(StrEnum):
    """Значения совпадают с `supervisor.validator.RejectReason` для прямого переиспользования."""

    NUMBER = "number_outside_context"
    NUMBER_WORD = "number_word_outside_context"
    ADDRESS = "address_outside_context"
    SERVICE = "service_outside_context"
    NAME = "name_outside_context"
    FOREIGN_TEXT = "foreign_text"


@dataclass(frozen=True)
class FactViolation:
    kind: FactViolationKind
    detail: str = ""


def is_number_word(tok: str) -> bool:
    return tok in _NUMBER_WORDS or bool(_NUMBER_WORD_RE.match(tok))


def sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip()
    return not before or before[-1] in ".!?"


def _stems_in(toks: list[str], stems: tuple[str, ...]) -> set[str]:
    return {s for s in stems for t in toks if t.startswith(s)}


def find_violation(
    text: str,
    *,
    allowed: str,
    check_services: bool = True,
    check_names: bool = True,
    check_foreign: bool = True,
) -> FactViolation | None:
    """Первое найденное «выдуманное» число/адрес/служба/имя или `None`.

    `allowed` — весь разрешённый контекст (черновик, сведения, опубликованные факты сценария),
    склеенный в одну строку. Проверка консервативна и однонаправленная: она ничего не решает
    о смысле текста, только о появлении новых сущностей, которых не было в контексте.
    """
    toks = matching.tokens(text)
    allowed_toks = set(matching.tokens(allowed))

    extra_digits = matching.digits(text) - matching.digits(allowed)
    if extra_digits:
        return FactViolation(FactViolationKind.NUMBER, ",".join(sorted(extra_digits)))

    new = [t for t in toks if t not in allowed_toks]
    if any(is_number_word(t) for t in new):
        return FactViolation(FactViolationKind.NUMBER_WORD)

    allowed_address = {ADDRESS_GROUPS[t] for t in allowed_toks if t in ADDRESS_GROUPS}
    if {ADDRESS_GROUPS[t] for t in toks if t in ADDRESS_GROUPS} - allowed_address:
        return FactViolation(FactViolationKind.ADDRESS)

    if check_services:
        extra_service = _stems_in(toks, SERVICE_STEMS) - _stems_in(
            matching.tokens(allowed), SERVICE_STEMS
        )
        if extra_service:
            return FactViolation(FactViolationKind.SERVICE)

    if check_foreign and _LATIN.search(text):
        if not set(_LATIN.findall(text)) <= set(_LATIN.findall(allowed)):
            return FactViolation(FactViolationKind.FOREIGN_TEXT)

    if check_names:
        for m in _CAPITALIZED.finditer(text):
            word = matching.normalize(m.group())
            if not sentence_start(text, m.start()) and not all(
                t in allowed_toks for t in matching.tokens(word)
            ):
                return FactViolation(FactViolationKind.NAME, m.group())

    return None
