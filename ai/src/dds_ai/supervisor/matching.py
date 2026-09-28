"""Правила M1: нормализация текста, сопоставление по основам слов, классы реплик.

Это заменяемая часть: на M3–M5 сопоставление доклада уточняется моделями (NLP-подсказки,
семантический оцениватель), но положительное совпадение по правилу остаётся проверяемым.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..text_norm import digits, normalize, tokens

__all__ = [
    "any_pattern",
    "clauses",
    "digits",
    "is_injection",
    "is_no",
    "is_question",
    "is_yes",
    "normalize",
    "pattern_matches",
    "tokens",
]

# Сокращения, после которых точка не завершает фразу: «ул. Тестовая, д. 12».
_ABBREV = {"ул", "д", "г", "пр", "пос", "кв", "корп", "стр", "пер", "просп", "мкр", "с", "т"}


def _stem_in(stem: str, toks: Iterable[str]) -> bool:
    if stem.endswith("$"):  # «пожар$» — слово целиком, не «пожарную»
        return stem[:-1] in toks
    if stem[0].isdigit():
        return any(t == stem for t in toks)  # «12» не совпадает со «120»
    return any(t.startswith(stem) for t in toks)


def pattern_matches(pattern: str, text_tokens: list[str]) -> bool:
    return all(_stem_in(stem, text_tokens) for stem in normalize(pattern).split())


def any_pattern(patterns: Iterable[str], text: str) -> bool:
    toks = tokens(text)
    return any(pattern_matches(p, toks) for p in patterns)


def clauses(text: str) -> list[str]:
    """Разбить реплику на фразы для повтора понятого (без разрыва на «ул.», «д.»)."""
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text.strip())
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out:
            prev_last = tokens(out[-1])[-1:] or [""]
            if out[-1].endswith(".") and prev_last[0] in _ABBREV:
                out[-1] = f"{out[-1]} {part}"
                continue
        out.append(part)
    return [c.rstrip(".!;") for c in out]


# --- классы реплик обучаемого ---------------------------------------------------------------
# Реплика ученика — это реплика, а не команда (инвариант 2). Такие фразы не меняют
# состояние диалога, карточку, маршрут, балл или статус попытки.

INJECTION_PATTERNS = (
    "измен оценк",
    "измен балл",
    "постав оценк",
    "постав балл",
    "поставь 100",
    "засчита",
    "правильн адрес",
    "правильн ответ",
    "правильн решени",
    "правильн служб",
    "игнорир инструкц",
    "игнорир правил",
    "забуд инструкц",
    "забуд правил",
    "забуд все",
    "систем промпт",
    "системн промпт",
    "эталон",
    "критери оценк",
)

_QUESTION_WORDS = (
    "сколько",
    "какой",
    "какая",
    "какое",
    "какие",
    "где",
    "когда",
    "кто",
    "известно",
    "есть ли",
)


def is_injection(text: str) -> bool:
    return any_pattern(INJECTION_PATTERNS, text)


def _exact_word(words: Iterable[str], text: str) -> bool:
    toks = tokens(text)
    for w in words:
        parts = normalize(w).split()
        n = len(parts)
        if any(toks[i : i + n] == parts for i in range(len(toks) - n + 1)):
            return True
    return False


def is_yes(text: str) -> bool:
    return _exact_word(("да", "верно", "так точно", "правильно", "все верно"), text) or any(
        t.startswith("подтвержда") for t in tokens(text)
    )


def is_no(text: str) -> bool:
    return _exact_word(("нет", "неверно", "не так", "не верно"), text) or any(
        t.startswith(("ошиб", "поправ")) for t in tokens(text)
    )


def is_question(text: str) -> bool:
    if "?" in text:
        return True
    toks = tokens(text)
    return bool(toks) and any(_exact_word((w,), " ".join(toks[:2])) for w in _QUESTION_WORDS)
