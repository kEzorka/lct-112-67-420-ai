"""Проверенный шаблон реплик руководителя — единственный генератор реплик на M1 (C-04)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Formatter

CONFIG = Path(__file__).resolve().parents[3] / "config"
DEFAULT_TEMPLATES = CONFIG / "supervisor.fallback.ru.json"

REQUIRED_KEYS = frozenset(
    {
        "listen.greeting",
        "listen.refusal",
        "listen.fact_known",
        "listen.fact_unknown",
        "listen.fact_none",
        "listen.fact_not_applicable",
        "listen.unknown_question",
        "listen.after_confirm",
        "clarify.essence",
        "clarify.location",
        "clarify.circumstances",
        "clarify.dds_decision",
        "clarify.requested_action",
        "clarify.correction",
        "read_back",
        "confirm_receipt",
    }
)
_ALLOWED_FIELDS = frozenset({"items", "label", "value"})


@dataclass(frozen=True)
class FallbackTemplates:
    version: str
    templates: dict[str, str]

    def __post_init__(self) -> None:
        missing = REQUIRED_KEYS - self.templates.keys()
        if missing:
            raise ValueError(f"fallback templates missing keys: {sorted(missing)}")
        for key, text in self.templates.items():
            fields = {f for _, f, _, _ in Formatter().parse(text) if f}
            if not fields <= _ALLOWED_FIELDS:
                raise ValueError(f"template {key} uses forbidden fields {sorted(fields)}")

    @classmethod
    def load(cls, path: Path = DEFAULT_TEMPLATES) -> FallbackTemplates:
        data = json.loads(path.read_text("utf-8"))
        return cls(version=data["template_version"], templates=dict(data["templates"]))

    def render(self, key: str, **fields: str) -> str:
        return self.templates[key].format(**fields)
