"""База знаний / RAG (6.10): версия корпуса и фрагмент со ссылкой на источник.

RAG — решение команды, не требование ТЗ (Miro 07); модуль заменяемый (Degradation.
GENERATORS_WITHOUT_RAG). Нормативная маршрутизация из этих фрагментов не берётся —
только из движка правил (D-016, C-03).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, Field

from .common import Contract, NonEmptyStr, VersionRef


class ApprovalStatus(StrEnum):
    """Статус утверждения источника/фрагмента (6.10)."""

    DRAFT = "draft"
    APPROVED = "approved"
    ARCHIVED = "archived"


class SourceDocument(Contract):
    """Метаданные исходного документа корпуса до чанкинга."""

    doc_id: NonEmptyStr
    title: NonEmptyStr
    source: NonEmptyStr = Field(description="Название источника, например 'инструкция АРМ-112'")
    section: str | None = None
    doc_version: NonEmptyStr
    published_at: AwareDatetime | None = None
    approval_status: ApprovalStatus
    synthetic: bool = Field(
        description="Исходники заказчика в репозиторий не входят; здесь — явно синтетический"
    )


class CorpusVersion(Contract):
    """Версия проиндексированного корпуса (входит в снимок версий сценария, D-003)."""

    name: NonEmptyStr = "knowledge_corpus"
    version: NonEmptyStr
    built_at: AwareDatetime
    document_count: int = Field(ge=0)
    fragment_count: int = Field(ge=0)

    @property
    def ref(self) -> VersionRef:
        return VersionRef(name=self.name, version=self.version)


class FragmentSource(Contract):
    """Ссылка фрагмента на исходный документ (обязательна — приёмка 6.10)."""

    doc_id: NonEmptyStr
    title: NonEmptyStr
    section: str | None = None
    source: NonEmptyStr
    approval_status: ApprovalStatus


class KnowledgeFragment(Contract):
    """Чанк корпуса с метаданными: источник, раздел, версия, дата, статус утверждения."""

    fragment_id: NonEmptyStr
    corpus: VersionRef
    source: FragmentSource
    chunk_index: int = Field(ge=0)
    text: NonEmptyStr
    created_at: AwareDatetime
