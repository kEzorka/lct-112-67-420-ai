"""Мок хранилища карточек: исходная карточка неизменна, ревизии только добавляются."""

from __future__ import annotations

from uuid import UUID

from ..contracts.card import CardRevisionRecord, IncidentCard


class ImmutableError(Exception):
    pass


class InMemoryCardStore:
    def __init__(self) -> None:
        self._sources: dict[UUID, IncidentCard] = {}
        self._revisions: dict[UUID, list[CardRevisionRecord]] = {}

    def put_source(self, card: IncidentCard) -> None:
        if card.card_id in self._sources:
            raise ImmutableError(f"source card {card.card_id} already stored")
        self._sources[card.card_id] = card
        self._revisions[card.card_id] = []

    def get_source(self, card_id: UUID) -> IncidentCard:
        return self._sources[card_id]

    def add_revision(self, revision: CardRevisionRecord) -> None:
        revs = self._revisions[revision.card_id]
        expected = len(revs) + 1
        if revision.revision != expected:
            raise ImmutableError(f"expected revision {expected}, got {revision.revision}")
        revs.append(revision)

    def revisions(self, card_id: UUID) -> list[CardRevisionRecord]:
        return list(self._revisions[card_id])
