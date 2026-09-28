"""Извлечение слотов из текста: адрес, число пострадавших (6.11).

Правило-ориентированная реализация для русского языка. Никаких служб не назначает
и не решает маршрутизацию (D-016) — только предлагает слоты и противоречия.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..contracts.nlp import ContradictionFlag, ExtractedSlot, MissingInfoFlag, SlotState, TextSpan

# Стемы слова «улица» (улица/улице/улицы/улицу) и др. — падежные окончания рус. языка.
ADDRESS_RE = re.compile(
    r"(?:ул\.|улиц\w*|пр-?кт|проспект\w*|пер\.|переул\w*|шоссе|бульвар\w*)\s*"
    r"([А-ЯЁа-яё0-9\-]+(?:\s[А-ЯЁа-яё0-9\-]+)*?)\s*,?\s*"
    r"(?:д\.?|дом\w*)\s*(\d+[а-яА-Я]?)"
    r"(?:\s*,?\s*(?:корп\.?|к\.?)\s*(\d+))?",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "два": 2,
    "две": 2,
    "двое": 2,
    "три": 3,
    "трое": 3,
    "четыре": 4,
    "четверо": 4,
    "пять": 5,
    "пятеро": 5,
    "шесть": 6,
    "шестеро": 6,
    "семь": 7,
    "семеро": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}
_NUMBER_WORD_ALT = "|".join(_NUMBER_WORDS)

# «пострада\w*» покрывает и глагол («пострадали»), и причастие («пострадавших»).
# Число может стоять и до, и после слова («два пострадавших» / «пострадавших двое»).
VICTIMS_RE = re.compile(
    rf"(?:(\d+|{_NUMBER_WORD_ALT})\s+(?:человек[а]?\s+)?(?:пострада\w*|ранен\w*|травмир\w*))"
    rf"|(?:(?:пострада\w*|ранен\w*|травмир\w*)\s+(\d+|{_NUMBER_WORD_ALT}))",
    re.IGNORECASE,
)
NO_VICTIMS_RE = re.compile(
    r"никто не пострадал\w*|пострадавших нет|без пострадавших", re.IGNORECASE
)

REQUIRED_SLOTS = ("address",)


@dataclass(frozen=True)
class _Mention:
    span: TextSpan
    value: int


def extract_address(text: str) -> ExtractedSlot:
    match = ADDRESS_RE.search(text)
    if match is None:
        return ExtractedSlot(name="address", state=SlotState.UNKNOWN)
    street = re.sub(r"\s+", " ", match.group(1)).strip().lower()
    house = match.group(2).lower()
    block = match.group(3)
    value = f"ул. {street}, д. {house}"
    if block:
        value += f", корп. {block}"
    return ExtractedSlot(
        name="address",
        state=SlotState.KNOWN,
        value=value,
        evidence=TextSpan(excerpt=match.group(0), start=match.start(), end=match.end()),
    )


def _victim_mentions(text: str) -> list[_Mention]:
    mentions: list[_Mention] = []
    for m in VICTIMS_RE.finditer(text):
        word = (m.group(1) or m.group(2)).lower()
        value = int(word) if word.isdigit() else _NUMBER_WORDS[word]
        mentions.append(_Mention(TextSpan(excerpt=m.group(0), start=m.start(), end=m.end()), value))
    for m in NO_VICTIMS_RE.finditer(text):
        mentions.append(_Mention(TextSpan(excerpt=m.group(0), start=m.start(), end=m.end()), 0))
    mentions.sort(key=lambda mm: mm.span.start)
    return mentions


def extract_victims(text: str) -> tuple[ExtractedSlot, ContradictionFlag | None]:
    mentions = _victim_mentions(text)
    if not mentions:
        return ExtractedSlot(name="victims_count", state=SlotState.UNKNOWN), None

    distinct = {m.value for m in mentions}
    if len(distinct) > 1:
        first_per_value: dict[int, _Mention] = {}
        for m in mentions:
            first_per_value.setdefault(m.value, m)
        evidence = tuple(first_per_value[v].span for v in sorted(first_per_value))
        slot = ExtractedSlot(name="victims_count", state=SlotState.UNKNOWN)
        contradiction = ContradictionFlag(
            slot="victims_count",
            description=(
                "разные упоминания числа пострадавших в тексте: "
                + ", ".join(f"{e.excerpt!r}" for e in evidence)
            ),
            evidence=evidence,
        )
        return slot, contradiction

    m = mentions[0]
    slot = ExtractedSlot(
        name="victims_count", state=SlotState.KNOWN, value=str(m.value), evidence=m.span
    )
    return slot, None


def missing_required(slots: list[ExtractedSlot]) -> tuple[MissingInfoFlag, ...]:
    by_name = {s.name: s for s in slots}
    out = []
    for name in REQUIRED_SLOTS:
        slot = by_name.get(name)
        if slot is None or slot.state is not SlotState.KNOWN:
            out.append(MissingInfoFlag(slot=name, description=f"не найдено в тексте: {name}"))
    return tuple(out)
