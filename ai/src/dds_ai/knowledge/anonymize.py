"""Обезличивание текста корпуса перед индексацией (6.10): телефоны, ФИО, адреса
физлиц, e-mail. Работает построчно/по тексту фрагмента, без внешних моделей.

Инвариант приёмки: телефон/ФИО из входного текста не должны находиться в чанках
индекса — проверяется тестами `test_knowledge_anonymize.py`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

PHONE_RE = re.compile(r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# «Фамилия И.О.» — самый частый вид ФИО в инструкциях и примерах.
_FIO_INITIALS_RE = re.compile(r"\b[А-ЯЁ][а-яё]+\s[А-ЯЁ]\.\s?[А-ЯЁ]\.")
# «Фамилия Имя Отчество» — отчество опознаётся по типовому суффиксу.
_FIO_FULL_RE = re.compile(
    r"\b[А-ЯЁ][а-яё]+\s[А-ЯЁ][а-яё]+\s"
    r"[А-ЯЁ][а-яё]*(?:ович|евич|ич|овна|евна|инична)\b"
)
# Улица/переулок/проспект + номер дома + номер квартиры — адрес проживания физлица,
# в отличие от места происшествия (там нет привязки к конкретной квартире).
ADDRESS_RE = re.compile(
    r"(?:ул\.|улица|пр-?кт|проспект|пер\.|переулок|мкр\.?)\s?"
    r"[А-ЯЁа-яё0-9\- ]{2,40}?,?\s?д\.?\s?\d+[а-яА-Я]?\s?,?\s?кв\.?\s?\d+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnonymizeHit:
    category: str
    original: str


_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", EMAIL_RE, "[EMAIL]"),
    ("phone", PHONE_RE, "[PHONE]"),
    ("address", ADDRESS_RE, "[ADDRESS]"),
    ("full_name", _FIO_FULL_RE, "[NAME]"),
    ("full_name", _FIO_INITIALS_RE, "[NAME]"),
)


def anonymize_text(text: str) -> tuple[str, tuple[AnonymizeHit, ...]]:
    """Заменить персональные данные на плейсхолдеры; вернуть текст и снятые совпадения.

    Правила применяются по очереди (email/phone/address раньше имён), поэтому уже
    заменённый фрагмент повторно не матчится — перекрытий между категориями нет.
    """
    hits: list[AnonymizeHit] = []
    out = text
    for category, pattern, placeholder in _RULES:
        out = _substitute(pattern, category, placeholder, out, hits)
    return out, tuple(hits)


def _substitute(
    pattern: re.Pattern[str],
    category: str,
    placeholder: str,
    text: str,
    hits: list[AnonymizeHit],
) -> str:
    def repl(match: re.Match[str]) -> str:
        hits.append(AnonymizeHit(category=category, original=match.group(0)))
        return placeholder

    return pattern.sub(repl, text)


AnonymizeFn = Callable[[str], tuple[str, tuple[AnonymizeHit, ...]]]
