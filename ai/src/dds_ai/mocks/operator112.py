"""Мок ИИ-оператора 112 на M1: карточка строится только из опубликованных фактов сценария.

Генерация вариаций формулировок — M4 (6.5). Здесь вариаций нет, seed фиксирован.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from ..contracts.card import CardField, CardGeneration, FieldOrigin, IncidentCard
from ..contracts.common import VersionRef
from ..contracts.scenario import Scenario

CARD_SCHEMA = VersionRef(name="card_schema", version="draft-1")


def build_source_card(scenario: Scenario, *, created_at: datetime, card_id: UUID | None = None):
    fields = {
        f.topic: CardField(state=f.state, raw=f.value, origin=FieldOrigin.OPERATOR_112)
        for f in scenario.published_facts
    }
    return IncidentCard(
        card_id=card_id or uuid4(),
        card_schema=CARD_SCHEMA,
        created_at=created_at,
        fields=fields,
        generation=CardGeneration(scenario=scenario.scenario, seed=0),
    )
