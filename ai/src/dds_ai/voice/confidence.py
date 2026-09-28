"""Сигнал неуверенности STT по критическим сущностям — адрес, числа, службы (6.2, инвариант 5).

Не изобретает confidence: если модель его не отдаёт (`TranscriptSegment.confidence is None`),
решение об неуверенности принимается только по структурным признакам искажения (например,
адрес без номера дома), а не по выдуманной вероятности. Флаг неуверенности — повод для
руководителя переспросить; сам по себе он не штраф и не «не проверено» (это решает оцениватель).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..ports import TranscriptSegment
from .hints import ADDRESS_MARKERS, SERVICE_NAMES

EntityKind = Literal["address", "number", "service"]

LOW_CONFIDENCE_THRESHOLD = 0.6

_NUMBER_RE = re.compile(r"\d+")
_HOUSE_NUMBER_RE = re.compile(r"\bд(?:ом)?\.?\s*\d+[а-яa-z]?\b", re.IGNORECASE)


@dataclass(frozen=True)
class UncertainEntity:
    kind: EntityKind
    text: str
    reason: Literal["low_confidence", "no_house_number"]
    segment_index: int


def _mentions_any(text: str, markers: Sequence[str]) -> bool:
    low = text.lower()
    return any(marker in low for marker in markers)


def detect_uncertain_entities(
    segments: Sequence[TranscriptSegment],
) -> tuple[UncertainEntity, ...]:
    """Помечает финальные сегменты с критическими сущностями, которым нельзя молча доверять."""
    out: list[UncertainEntity] = []
    for index, segment in enumerate(segments):
        if not segment.final:
            continue
        low_confidence = (
            segment.confidence is not None and segment.confidence < LOW_CONFIDENCE_THRESHOLD
        )
        has_address = _mentions_any(segment.text, ADDRESS_MARKERS)
        has_number = bool(_NUMBER_RE.search(segment.text))
        has_service = _mentions_any(segment.text, SERVICE_NAMES)

        if has_address:
            if low_confidence:
                out.append(UncertainEntity("address", segment.text, "low_confidence", index))
            elif not _HOUSE_NUMBER_RE.search(segment.text):
                out.append(UncertainEntity("address", segment.text, "no_house_number", index))
        if has_number and low_confidence:
            out.append(UncertainEntity("number", segment.text, "low_confidence", index))
        if has_service and low_confidence:
            out.append(UncertainEntity("service", segment.text, "low_confidence", index))
    return tuple(out)
