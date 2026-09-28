"""Индекс корпуса и поиск (6.10): запрос + фильтр версии → фрагменты со ссылкой на источник.

Embedding идёт только через порт `Embedder` и `FaultInjector.call()`. Отказ embedder —
`ComponentFailure`, попадает в журнал `FaultInjector.log`; деградация задана в
`faults.DEGRADATION[EMBEDDER] == Degradation.GENERATORS_WITHOUT_RAG` — генераторы (6.5)
продолжают работать на явно переданных материалах, а не на молчаливо пустом поиске.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts.common import VersionRef
from ..contracts.events import ComponentName
from ..contracts.knowledge import KnowledgeFragment
from ..faults import FaultInjector
from ..ports import Embedder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class SearchHit:
    fragment: KnowledgeFragment
    score: float


class KnowledgeIndex:
    """Индекс одной версии корпуса. Смена версии — новый индекс; этот не меняется (6.10)."""

    def __init__(
        self,
        corpus: VersionRef,
        fragments: list[KnowledgeFragment],
        embedder: Embedder,
        injector: FaultInjector,
    ):
        for f in fragments:
            if f.corpus != corpus:
                raise ValueError(f"fragment {f.fragment_id} belongs to {f.corpus}, not {corpus}")
        self.corpus = corpus
        self._fragments = list(fragments)
        self._embedder = embedder
        self._injector = injector
        self._vectors = injector.call(
            ComponentName.EMBEDDER,
            lambda: embedder.embed([f.text for f in self._fragments]),
        )

    def search(
        self, query: str, *, corpus_version: VersionRef | None = None, top_k: int = 5
    ) -> list[SearchHit]:
        """Поиск в этой версии корпуса. `corpus_version`, если задан, должен совпасть —
        иначе вызывающий перепутал версию индекса и не получит фрагменты чужой версии."""
        if corpus_version is not None and corpus_version != self.corpus:
            return []
        (query_vec,) = self._injector.call(
            ComponentName.EMBEDDER, lambda: self._embedder.embed([query])
        )
        scored = [
            SearchHit(fragment=f, score=_cosine(query_vec, v))
            for f, v in zip(self._fragments, self._vectors, strict=True)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return [h for h in scored if h.score > 0][:top_k]
