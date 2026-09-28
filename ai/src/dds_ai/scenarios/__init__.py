"""ИИ-оператор 112 и генерация сценариев (6.5): карточка, черновики, структурированный
эталон, жизненный цикл и gate C-03. См. модули `card_generation`, `draft_generation`,
`reference_generation`, `lifecycle`, `coverage`.
"""

from __future__ import annotations

from .card_generation import generate_card, generate_card_with_llm
from .coverage import CoverageCell, coverage_cells
from .draft_generation import generate_draft
from .lifecycle import (
    GatePublishError,
    LifecycleError,
    approve,
    archive,
    assert_assignable,
    check_publish_gate,
    publish,
    revise,
    submit_for_review,
    validate_structure,
)
from .reference_generation import generate_reference

__all__ = [
    "CoverageCell",
    "GatePublishError",
    "LifecycleError",
    "approve",
    "archive",
    "assert_assignable",
    "check_publish_gate",
    "coverage_cells",
    "generate_card",
    "generate_card_with_llm",
    "generate_draft",
    "generate_reference",
    "publish",
    "revise",
    "submit_for_review",
    "validate_structure",
]
