"""Локальная проверка грамотности ручных правок карточки ([ТЗ], D-034, инвариант 10).

Только правила, без модели и без сети: обнаруживает механические ошибки текста (повтор
слова, пробелы вокруг пунктуации, ряд знаков препинания, строчная буква после точки).
Это не полная проверка орфографии/пунктуации русского языка — редкие и стилистические
ошибки не покрыты. Замечание содержит фрагмент и предлагаемое исправление; текст ученика
не меняется — `suggestion` не применяется автоматически нигде в коде.

Офлайн-инструмент вроде LanguageTool не подключён по умолчанию: локальный сервер Java
недоступен в базовой поставке пакета и не проверен без сети (инвариант 8). Функция
`check_text` — единственная точка вызова; расширение источником замечаний (LanguageTool
локально) не меняет её контракт.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_REPEATED_WORD = re.compile(r"\b([а-яё]{2,})\s+\1\b", re.IGNORECASE)
_DOUBLE_SPACE = re.compile(r" {2,}")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.!?;:])")
_NO_SPACE_AFTER_PUNCT = re.compile(r"([,.!?;:])([А-ЯЁа-яёA-Za-z])")
_REPEATED_PUNCT = re.compile(r"([,.!?;:])\1+")
_LOWER_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([а-яё])")


class IssueKind(StrEnum):
    REPEATED_WORD = "repeated_word"
    DOUBLE_SPACE = "double_space"
    SPACE_BEFORE_PUNCT = "space_before_punct"
    NO_SPACE_AFTER_PUNCT = "no_space_after_punct"
    REPEATED_PUNCT = "repeated_punct"
    LOWER_SENTENCE_START = "lower_sentence_start"


# Только «настоящие» ошибки (не косметика) идут в счёт критерия manual.grammar (рабочая
# конвенция команды, не пункт W-01: грамматика в реестре решений порогов не имеет).
COUNTS_AGAINST_SCORE = frozenset({IssueKind.REPEATED_WORD, IssueKind.LOWER_SENTENCE_START})

_SUGGESTIONS = {
    IssueKind.REPEATED_WORD: "убрать повтор слова",
    IssueKind.DOUBLE_SPACE: "один пробел",
    IssueKind.SPACE_BEFORE_PUNCT: "убрать пробел перед знаком препинания",
    IssueKind.NO_SPACE_AFTER_PUNCT: "пробел после знака препинания",
    IssueKind.REPEATED_PUNCT: "один знак препинания",
    IssueKind.LOWER_SENTENCE_START: "заглавная буква в начале предложения",
}


@dataclass(frozen=True)
class GrammarIssue:
    kind: IssueKind
    start: int
    end: int
    fragment: str
    suggestion: str

    @property
    def counts_against_score(self) -> bool:
        return self.kind in COUNTS_AGAINST_SCORE


def _context(text: str, start: int, end: int, pad: int = 8) -> str:
    return text[max(0, start - pad) : min(len(text), end + pad)]


def check_text(text: str) -> list[GrammarIssue]:
    """Найти механические ошибки в тексте. Порядок находок — по позиции в тексте."""
    if not text.strip():
        return []
    found: list[GrammarIssue] = []
    for kind, pattern in (
        (IssueKind.REPEATED_WORD, _REPEATED_WORD),
        (IssueKind.DOUBLE_SPACE, _DOUBLE_SPACE),
        (IssueKind.SPACE_BEFORE_PUNCT, _SPACE_BEFORE_PUNCT),
        (IssueKind.NO_SPACE_AFTER_PUNCT, _NO_SPACE_AFTER_PUNCT),
        (IssueKind.REPEATED_PUNCT, _REPEATED_PUNCT),
    ):
        for m in pattern.finditer(text):
            found.append(
                GrammarIssue(
                    kind=kind,
                    start=m.start(),
                    end=m.end(),
                    fragment=_context(text, m.start(), m.end()),
                    suggestion=_SUGGESTIONS[kind],
                )
            )
    for m in _LOWER_SENTENCE_START.finditer(text):
        start, end = m.start(1), m.end(1)
        found.append(
            GrammarIssue(
                kind=IssueKind.LOWER_SENTENCE_START,
                start=start,
                end=end,
                fragment=_context(text, start, end),
                suggestion=_SUGGESTIONS[IssueKind.LOWER_SENTENCE_START],
            )
        )
    found.sort(key=lambda i: i.start)
    return found
