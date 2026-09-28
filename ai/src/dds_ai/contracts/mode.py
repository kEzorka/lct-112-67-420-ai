"""Режимы обучения и трек попытки (D-041, 6.7).

«Делай как я» и самостоятельная работа используют одну модель попытки; результат с
поддержкой не смешивается с самостоятельным (инвариант — критерий приёмки D-041).
"""

from __future__ import annotations

from enum import StrEnum


class TrainingMode(StrEnum):
    """Режим, выбранный до старта попытки."""

    GUIDED = "guided"  # «делай как я»: демонстрация и пошаговые подсказки
    INDEPENDENT = "independent"  # самостоятельный: подсказки о правильном действии скрыты


class AttemptTrack(StrEnum):
    """Фактический трек попытки для профиля и рейтинга — вычисляется из режима и событий.

    `help_requested` в самостоятельном режиме переводит попытку в `supported`: она перестаёт
    быть самостоятельной, но это не «делай как я» (там не было демонстрации, была честная
    попытка без подсказок до запроса помощи).
    """

    GUIDED = "guided"
    INDEPENDENT = "independent"
    SUPPORTED = "supported"

    @property
    def counts_toward_independent_rating(self) -> bool:
        """Только полностью самостоятельный результат идёт в самостоятельный рейтинг (D-041)."""
        return self is AttemptTrack.INDEPENDENT
