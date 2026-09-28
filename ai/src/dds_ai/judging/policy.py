"""Конфиг оценивателя: каким критериям `field_fact` модель может самостоятельно ставить `failed`.

Решение F-2 (волна 3, `docs/ai/progress.md`): пока согласие семантического оценивателя с
рабочей разметкой ниже порога W-03 (85%, `docs/ai/calibration-report.md` — на Qwen2.5-1.5B
59%), модель одна не выносит `failed` для критериев `card.circumstances`/`manual.additions`
(`field_fact`) — уверенное `none` уходит в `not_checked` с объяснением «ждёт эксперта», как уже
устроено для `voice.facts_transferred` (`conversation_coverage` структурно не может дать
`failed` вовсе, см. `semantic_judge.py`). Какие критерии всё же разрешено самостоятельно
проваливать — конфигурация команды, не код: список пуст по умолчанию.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[3] / "config"
DEFAULT_POLICY = CONFIG / "judge.policy.json"


@dataclass(frozen=True)
class JudgePolicy:
    failable_criteria: frozenset[str] = frozenset()

    @classmethod
    def load(cls, path: Path = DEFAULT_POLICY) -> JudgePolicy:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text("utf-8"))
        return cls(failable_criteria=frozenset(data.get("failable_criteria", [])))
