"""Режим попытки и доступность подсказок о правильном действии (D-041, 6.7).

Подсказка не раскрывает эталон сверх того, что разрешено режимом: в «делай как я» —
демонстрация и пошаговые подсказки доступны всегда; в самостоятельном режиме — только
после явного запроса помощи (`HelpRequested`), который переводит попытку в «с поддержкой»
и исключает её из самостоятельного рейтинга.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts.events import AttemptEvent, HelpRequested
from ..contracts.mode import AttemptTrack, TrainingMode


class HintDenied(Exception):
    """Подсказка о правильном действии недоступна в этом режиме/на этом этапе (D-041)."""


def attempt_track(mode: TrainingMode, events: Sequence[AttemptEvent]) -> AttemptTrack:
    """Трек попытки: guided остаётся guided; independent + help_requested → supported."""
    if mode is TrainingMode.GUIDED:
        return AttemptTrack.GUIDED
    if any(isinstance(e, HelpRequested) for e in events):
        return AttemptTrack.SUPPORTED
    return AttemptTrack.INDEPENDENT


def can_show_guided_hint(mode: TrainingMode) -> bool:
    """Демонстрация и пошаговые подсказки «делай как я» — доступны только в guided.

    В самостоятельном режиме подсказка о правильном действии не показывается до завершения
    попытки вообще (D-041) — запрос помощи (`request_help`) меняет только классификацию
    попытки (трек `supported`, исключение из самостоятельного рейтинга), не открывает
    подсказки во время самой попытки. Разбор ошибок после завершения (`hints.explain_errors`)
    доступен в обоих режимах одинаково — это не подсказка во время попытки.
    """
    return mode is TrainingMode.GUIDED
