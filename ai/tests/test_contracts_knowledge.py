"""Контракты корпуса и фрагмента (6.10): версия корпуса, фрагмент со ссылкой на источник."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from dds_ai.contracts.common import VersionRef
from dds_ai.contracts.knowledge import (
    ApprovalStatus,
    CorpusVersion,
    FragmentSource,
    KnowledgeFragment,
    SourceDocument,
)

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def test_corpus_version_ref_matches_name_and_version():
    corpus = CorpusVersion(version="v3", built_at=T0, document_count=2, fragment_count=5)
    assert corpus.ref == VersionRef(name="knowledge_corpus", version="v3")


def test_fragment_requires_source_and_corpus_version():
    with pytest.raises(ValidationError):
        KnowledgeFragment(
            fragment_id="f1",
            source=FragmentSource(
                doc_id="d1",
                title="t",
                source="s",
                approval_status=ApprovalStatus.APPROVED,
            ),
            chunk_index=0,
            text="текст",
            created_at=T0,
        )  # нет corpus


def test_fragment_is_immutable():
    fragment = KnowledgeFragment(
        fragment_id="f1",
        corpus=VersionRef(name="knowledge_corpus", version="v1"),
        source=FragmentSource(
            doc_id="d1", title="t", source="s", approval_status=ApprovalStatus.APPROVED
        ),
        chunk_index=0,
        text="текст",
        created_at=T0,
    )
    with pytest.raises(ValidationError):
        fragment.chunk_index = 1


def test_source_document_marks_synthetic_explicitly():
    doc = SourceDocument(
        doc_id="d1",
        title="t",
        source="s",
        doc_version="v1",
        approval_status=ApprovalStatus.DRAFT,
        synthetic=True,
    )
    assert doc.synthetic is True
    with pytest.raises(ValidationError):
        SourceDocument(
            doc_id="d1",
            title="t",
            source="s",
            doc_version="v1",
            approval_status=ApprovalStatus.DRAFT,
        )  # synthetic обязателен явно
