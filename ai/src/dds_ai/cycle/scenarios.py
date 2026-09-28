"""Загрузка сценариев. На M1 — только явно синтетические из `ai/data/synthetic/`."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts.scenario import Provenance, Scenario

SYNTHETIC_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(json.loads(path.read_text("utf-8")))


def load_synthetic() -> dict[str, Scenario]:
    out = {}
    for path in sorted(SYNTHETIC_DIR.glob("*.json")):
        s = load_scenario(path)
        if s.provenance is not Provenance.SYNTHETIC:
            raise ValueError(f"{path.name}: non-synthetic scenario in the synthetic set")
        out[s.scenario.name] = s
    return out
