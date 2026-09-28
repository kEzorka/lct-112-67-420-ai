"""Режим попытки и доступность подсказок о правильном действии (D-041, G-1, 6.7).

Подсказка не раскрывает эталон сверх того, что разрешено режимом: в «делай как я» —
демонстрация и пошаговые подсказки доступны всегда; в самостоятельном режиме — только
после явного запроса помощи (`HelpRequested`), который сразу открывает подсказки (иначе
кнопка запроса помощи бесполезна — решение G-1) и переводит попытку в «с поддержкой»,
исключая её из самостоятельного рейтинга.
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


def can_show_guided_hint(mode: TrainingMode, events: Sequence[AttemptEvent] = ()) -> bool:
    """Подсказка о правильном действии доступна в guided всегда и в independent — сразу
    после запроса помощи (G-1: кнопка запроса помощи иначе бесполезна).

    До запроса помощи в самостоятельном режиме подсказка недоступна (D-041). Разбор ошибок
    после завершения (`hints.explain_errors`) доступен в обоих режимах одинаково — это не
    подсказка во время попытки и этой функции не касается.
    """
    return attempt_track(mode, events) is not AttemptTrack.INDEPENDENT
