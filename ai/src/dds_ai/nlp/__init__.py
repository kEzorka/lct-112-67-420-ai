"""NLP-извлечение признаков (6.11): подсказка (`ML hints`) движку правил и оценивателю.

Служб не назначает; финальный тип, ЕКП и маршрут — только через `mocks/routing.py`.
"""

from .extract import classify_incident_types, extract_features, extract_tags
from .slots import extract_address, extract_victims

__all__ = [
    "classify_incident_types",
    "extract_address",
    "extract_features",
    "extract_tags",
    "extract_victims",
]
