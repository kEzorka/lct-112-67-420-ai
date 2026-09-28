"""Строгая схема выхода модели-оценивателя (6.6). Невалидный JSON/лишние поля/неизвестный
`reason_code`/цитата, не найденная дословно в проверяемом тексте, — `InvalidOutput` → выше по
стеку (`evaluation.judge_criterion`) превращается в `not_checked`, никогда не в 0 (инвариант 4).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ..contracts.common import Contract, NonEmptyStr


class MatchLevel(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"
    UNCLEAR = "unclear"  # неразборчиво/повреждено — не штраф (инвариант 4, STT-устойчивость)


REASON_CODES = frozenset(
    {
        "exact_match",
        "paraphrase_ok",
        "incomplete",
        "missing",
        "wrong_value",
        "ambiguous_transcript",
    }
)


class FieldVerdict(Contract):
    match: MatchLevel
    quote: str = Field(default="", description="Дословная подстрока проверяемого текста")
    reason_code: NonEmptyStr


class ItemVerdict(Contract):
    item_id: NonEmptyStr
    match: MatchLevel
    quote: str = Field(default="", description="Дословная подстрока транскрипта")


class ConversationVerdict(Contract):
    items: tuple[ItemVerdict, ...]
