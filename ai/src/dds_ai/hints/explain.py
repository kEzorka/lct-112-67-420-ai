"""Объяснение ошибок после завершения попытки (Q&A 8, D-041, D-045).

Цепочка: конкретная ошибка → доказательство → упражнение для исправления. Строится только
из замечаний, которые действительно являются ошибкой ученика (`Remark.counts_as_error`) —
технические сбои и неопределённость источника сюда не попадают (инвариант 4).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..contracts.remarks import ErrorExplanation, Remark

CONFIG = Path(__file__).resolve().parents[3] / "config"
DEFAULT_EXERCISES = CONFIG / "exercises.ru.json"


def load_exercises(path: Path = DEFAULT_EXERCISES) -> dict[str, str]:
    data = json.loads(path.read_text("utf-8"))
    by_type: dict[str, str] = dict(data["by_remark_type"])
    by_type["default"] = data["default"]
    return by_type


def explain_errors(
    remarks: Sequence[Remark], *, exercises: Mapping[str, str] | None = None
) -> tuple[ErrorExplanation, ...]:
    """Только замечания-ошибки ученика; каждое — с упражнением по типу замечания."""
    ex = dict(exercises) if exercises is not None else load_exercises()
    default = ex.get("default", "Обсудите это замечание с преподавателем.")
    out = []
    for r in remarks:
        if not r.counts_as_error:
            continue
        out.append(
            ErrorExplanation(
                remark_id=r.remark_id,
                error=r.text,
                evidence=r.evidence,
                exercise=ex.get(r.type.value, default),
            )
        )
    return tuple(out)
