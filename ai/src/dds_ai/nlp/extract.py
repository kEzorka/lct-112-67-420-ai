"""Базовая (правило-ориентированная) реализация NLP-извлечения признаков (6.11).

Вход: текст описания / SMS / транскрипт. Выход — `NlpExtraction` (ML hints): top-k
типов происшествия, теги (в т.ч. существенные обстоятельства), слоты, пропуски и
противоречия. Confidence у правил всегда `None` (инвариант 5). Финальный тип, ЕКП и
службы этот модуль не выбирает — решает только детерминированный движок маршрутизации.
"""

from __future__ import annotations

from ..contracts.nlp import IncidentTypeHint, NlpExtraction
from .lexicon import INCIDENT_KEYWORDS, TAG_KEYWORDS
from .slots import extract_address, extract_victims, missing_required


def classify_incident_types(text: str, *, top_k: int = 3) -> tuple[IncidentTypeHint, ...]:
    lower = text.lower()
    scores = {
        incident_type: sum(lower.count(keyword) for keyword in keywords)
        for incident_type, keywords in INCIDENT_KEYWORDS.items()
    }
    ranked = sorted((t for t, s in scores.items() if s > 0), key=lambda t: (-scores[t], t))
    return tuple(IncidentTypeHint(incident_type=t) for t in ranked[:top_k])


def extract_tags(text: str) -> tuple[str, ...]:
    lower = text.lower()
    return tuple(
        tag for tag, keywords in TAG_KEYWORDS.items() if any(kw in lower for kw in keywords)
    )


def extract_features(text: str) -> NlpExtraction:
    """Базовая реализация на правилах. Модельный вариант подключается через тот же
    контракт `NlpExtraction` (поле `model_ref` заполняется, confidence может появиться)."""
    address_slot = extract_address(text)
    victims_slot, victims_contradiction = extract_victims(text)
    slots = (address_slot, victims_slot)

    missing = missing_required(list(slots))
    contradictions = (victims_contradiction,) if victims_contradiction else ()

    return NlpExtraction(
        top_k=classify_incident_types(text),
        tags=extract_tags(text),
        slots=slots,
        missing=missing,
        contradictions=contradictions,
        needs_clarification=bool(missing) or bool(contradictions),
        model_ref=None,
    )
