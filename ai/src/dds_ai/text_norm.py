"""Базовая нормализация и токенизация текста — без зависимостей от других модулей пакета.

Вынесено из `supervisor/matching.py`, чтобы им мог пользоваться `facts_guard` (E-5) без
циклического импорта (facts_guard используется и супервизором, и генерацией карточки, и
семантическим оценивателем).
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\d+[а-яa-z]?|[а-яa-z]+")
_DIGITS = re.compile(r"\d+")


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def digits(text: str) -> set[str]:
    return set(_DIGITS.findall(text))
