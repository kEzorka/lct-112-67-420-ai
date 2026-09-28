"""Профиль подготовки и рекомендации (6.9, D-044, D-045, C-07, C-08)."""

from .build import DEFAULT_MIN_SEGMENT_ATTEMPTS, AttemptRecord, build_profile
from .recommend import MIN_ATTEMPTS_FOR_LEVEL_UP, apply_teacher_action, mark_superseded, recommend

__all__ = [
    "DEFAULT_MIN_SEGMENT_ATTEMPTS",
    "MIN_ATTEMPTS_FOR_LEVEL_UP",
    "AttemptRecord",
    "apply_teacher_action",
    "build_profile",
    "mark_superseded",
    "recommend",
]
