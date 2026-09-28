"""Нормализация текста перед синтезом речи (6.3): адреса, номера домов, числа, аббревиатуры служб.

Правило-ориентированная (не ML), без внешних зависимостей — работает в базовом пакете.
Не меняет факты и не исправляет текст диспетчера/руководителя (инвариант 10): применяется
только к репликам, которые сам ИИ-контур передаёт в TTS, не к вводу ученика.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .hints import KNOWN_SERVICE_ACRONYMS
from .numerals import num2ru

_LETTER_NAMES: Mapping[str, str] = {
    "а": "а",
    "б": "бэ",
    "в": "вэ",
    "г": "гэ",
    "д": "дэ",
    "е": "е",
    "ё": "ё",
    "ж": "жэ",
    "з": "зэ",
    "и": "и",
    "й": "и краткое",
    "к": "ка",
    "л": "эль",
    "м": "эм",
    "н": "эн",
    "о": "о",
    "п": "пэ",
    "р": "эр",
    "с": "эс",
    "т": "тэ",
    "у": "у",
    "ф": "эф",
    "х": "ха",
    "ц": "цэ",
    "ч": "че",
    "ш": "ша",
    "щ": "ща",
    "ъ": "",
    "ы": "ы",
    "ь": "",
    "э": "э",
    "ю": "ю",
    "я": "я",
}

# Раскрытие адресных сокращений до слов; "(?=\d)" — только перед номером,
# чтобы не разворачивать "д." в середине обычного слова.
_ADDRESS_ABBREVIATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bул\.\s*", re.IGNORECASE), "улица "),
    (re.compile(r"\bпер\.\s*", re.IGNORECASE), "переулок "),
    (re.compile(r"\bпр-?кт\.?\s*", re.IGNORECASE), "проспект "),
    (re.compile(r"\bпр-?т\.?\s*", re.IGNORECASE), "проспект "),
    (re.compile(r"\bш\.\s*", re.IGNORECASE), "шоссе "),
    (re.compile(r"\bпл\.\s*", re.IGNORECASE), "площадь "),
    (re.compile(r"\bнаб\.\s*", re.IGNORECASE), "набережная "),
    (re.compile(r"\bбульв?\.\s*", re.IGNORECASE), "бульвар "),
    (re.compile(r"\bмкр\.?\s*", re.IGNORECASE), "микрорайон "),
    (re.compile(r"\bд\.\s*(?=\d)", re.IGNORECASE), "дом "),
    (re.compile(r"\bкорп\.\s*(?=\d)", re.IGNORECASE), "корпус "),
    (re.compile(r"\bкв\.\s*(?=\d)", re.IGNORECASE), "квартира "),
    (re.compile(r"\bэт\.\s*(?=\d)", re.IGNORECASE), "этаж "),
)

# Номер дома с литерой: "9А", "12Б" → "девять а". Требует, чтобы после буквы не
# шёл ещё один буквенный символ (иначе это начало обычного слова, а не литера).
_NUM_WITH_LETTER = re.compile(r"(?<!\w)(\d+)\s*([А-Яа-яA-Za-z])(?!\w)")
_NUM = re.compile(r"\d+")
_ACRONYM = re.compile(r"[А-Яа-я]{2,6}")


def _spell_out(word: str) -> str:
    letters = [_LETTER_NAMES.get(ch.lower(), ch.lower()) for ch in word]
    return "-".join(letter for letter in letters if letter)


def _replace_num_with_letter(match: re.Match[str]) -> str:
    number = num2ru(int(match.group(1)))
    letter = _LETTER_NAMES.get(match.group(2).lower(), match.group(2))
    return f"{number} {letter}".strip()


def _replace_num(match: re.Match[str]) -> str:
    return num2ru(int(match.group()))


def _replace_acronyms(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        word = match.group()
        if word.lower() in KNOWN_SERVICE_ACRONYMS:
            return _spell_out(word)
        return word

    return _ACRONYM.sub(repl, text)


def normalize_for_tts(text: str) -> str:
    """Готовит реплику к синтезу: разворачивает сокращения адреса, числа и аббревиатуры служб."""
    result = text
    for pattern, replacement in _ADDRESS_ABBREVIATIONS:
        result = pattern.sub(replacement, result)
    result = _NUM_WITH_LETTER.sub(_replace_num_with_letter, result)
    result = _NUM.sub(_replace_num, result)
    result = _replace_acronyms(result)
    return re.sub(r"\s+", " ", result).strip()
