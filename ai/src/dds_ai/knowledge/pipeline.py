"""Pipeline базы знаний (6.10): загрузка → валидация → нормализация → обезличивание →
контроль качества → версия корпуса → чанкинг → метаданные → индекс.

`build_corpus()` не требует `Embedder` — версия корпуса и фрагменты доступны, даже
если embedder отказал (Degradation.GENERATORS_WITHOUT_RAG); поисковый индекс строит
отдельно `build_index()`, только он ходит к embedder через `FaultInjector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..contracts.common import VersionRef
from ..contracts.knowledge import CorpusVersion, FragmentSource, KnowledgeFragment
from ..faults import FaultInjector
from ..ports import Embedder
from .anonymize import AnonymizeHit, anonymize_text
from .chunking import chunk_text
from .documents import load_documents
from .index import KnowledgeIndex
from .normalize import normalize_text


@dataclass(frozen=True)
class QualityReport:
    """Контроль качества: сколько и каких персональных данных снято по документам."""

    hits_by_doc: dict[str, tuple[AnonymizeHit, ...]]

    @property
    def total_hits(self) -> int:
        return sum(len(hits) for hits in self.hits_by_doc.values())


@dataclass(frozen=True)
class BuiltCorpus:
    version: CorpusVersion
    fragments: tuple[KnowledgeFragment, ...]
    quality: QualityReport


def build_corpus(source_dir: Path, *, version: str, now: datetime | None = None) -> BuiltCorpus:
    docs = load_documents(source_dir)
    built_at = now or datetime.now(UTC)
    corpus_ref = VersionRef(name="knowledge_corpus", version=version)

    fragments: list[KnowledgeFragment] = []
    hits_by_doc: dict[str, tuple[AnonymizeHit, ...]] = {}
    for doc in docs:
        clean_text, hits = anonymize_text(normalize_text(doc.text))
        hits_by_doc[doc.meta.doc_id] = hits
        source = FragmentSource(
            doc_id=doc.meta.doc_id,
            title=doc.meta.title,
            section=doc.meta.section,
            source=doc.meta.source,
            approval_status=doc.meta.approval_status,
        )
        for index, chunk in enumerate(chunk_text(clean_text)):
            fragments.append(
                KnowledgeFragment(
                    fragment_id=f"{doc.meta.doc_id}::{version}::{index}",
                    corpus=corpus_ref,
                    source=source,
                    chunk_index=index,
                    text=chunk,
                    created_at=built_at,
                )
            )

    corpus = CorpusVersion(
        version=version,
        built_at=built_at,
        document_count=len(docs),
        fragment_count=len(fragments),
    )
    return BuiltCorpus(
        version=corpus, fragments=tuple(fragments), quality=QualityReport(hits_by_doc)
    )


def build_index(corpus: BuiltCorpus, embedder: Embedder, injector: FaultInjector) -> KnowledgeIndex:
    """Построить индекс через `Embedder`/`FaultInjector`. Сбой embedder поднимает
    `ComponentFailure` — вызывающий переходит на `Degradation.GENERATORS_WITHOUT_RAG`
    и использует `corpus.fragments` напрямую, без поиска."""
    return KnowledgeIndex(corpus.version.ref, list(corpus.fragments), embedder, injector)
