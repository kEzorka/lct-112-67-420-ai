"""Сложность: вектор факторов, детерминированный расчёт, предложение с объяснением (6.8, D-043)."""

from .band import band_for, load_bands
from .compute import contribution_of, load_weights, score
from .propose import propose_vector

__all__ = [
    "band_for",
    "contribution_of",
    "load_bands",
    "load_weights",
    "propose_vector",
    "score",
]
