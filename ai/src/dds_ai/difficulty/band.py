"""Границы сложности (low/medium/high) — публикует и утверждает преподаватель (D-043).

Рабочие пороги команды в `ai/config/difficulty.json::bands`, не пункт W-03: заменяются,
когда преподаватель утвердит границы для конкретного набора сценариев.
"""

from __future__ import annotations

import json
from pathlib import Path

from .compute import DEFAULT_WEIGHTS as DEFAULT_DIFFICULTY_CONFIG

Band = tuple[str, float]


def load_bands(path: Path = DEFAULT_DIFFICULTY_CONFIG) -> tuple[Band, ...]:
    data = json.loads(path.read_text("utf-8"))
    bands = tuple((b["band"], float(b["min_total"])) for b in data["bands"])
    ordered = tuple(sorted(bands, key=lambda b: b[1]))
    if ordered != bands:
        raise ValueError("bands must be listed in ascending min_total order")
    return bands


def band_for(total: float, bands: tuple[Band, ...] | None = None) -> str:
    """Наибольшая граница, чей `min_total` не превышает сумму (bands по возрастанию)."""
    bands = bands if bands is not None else load_bands()
    if not bands:
        raise ValueError("no bands configured")
    chosen = bands[0][0]
    for name, minimum in bands:
        if total >= minimum:
            chosen = name
    return chosen
