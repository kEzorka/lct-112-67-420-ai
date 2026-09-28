"""Подсказки и режимы обучения (6.7, D-041)."""

from .explain import explain_errors, load_exercises
from .track import HintDenied, attempt_track, can_show_guided_hint

__all__ = [
    "HintDenied",
    "attempt_track",
    "can_show_guided_hint",
    "explain_errors",
    "load_exercises",
]
