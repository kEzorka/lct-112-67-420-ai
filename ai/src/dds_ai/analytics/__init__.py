"""Аналитика типичных ошибок группы для преподавателя (6.9, [ТЗ])."""

from .group_errors import AttemptErrorRecord, build_group_error_report

__all__ = ["AttemptErrorRecord", "build_group_error_report"]
