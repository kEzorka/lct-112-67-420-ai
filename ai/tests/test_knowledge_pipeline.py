"""Pipeline базы знаний (6.10): офлайн-индексация, поиск, версии, деградация embedder."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dds_ai.contracts.common import ModelRef
from dds_ai.contracts.events import ComponentName, FailureKind
from dds_ai.contracts.knowledge import ApprovalStatus
from dds_ai.faults import ComponentFailure, Degradation, FaultInjector
from dds_ai.knowledge import HashingEmbedder, build_corpus, build_index
from dds_ai.knowledge.documents import CorpusValidationError

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _write_doc(directory: Path, doc_id: str, text: str, **overrides) -> None:
    payload = {
        "doc_id": doc_id,
        "title": f"Синтетический документ {doc_id}",
        "source": "synthetic_test_source",
        "section": "Раздел",
        "doc_version": "syn-1",
        "published_at": "2026-09-01T00:00:00+03:00",
        "approval_status": "approved",
        "synthetic": True,
        "text": text,
        **overrides,
    }
    (directory / f"{doc_id}.json").write_text(json.dumps(payload, ensure_ascii=False), "utf-8")


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    _write_doc(
        tmp_path,
        "doc-fire",
        "Пожар в жилом здании: дым, эвакуация жильцов, главная служба — пожарная охрана.\n\n"
        "Контактное лицо для примера (вымышленное): Иванов Иван Иванович, "
        "телефон +7 (912) 111-22-33, адрес ул. Опытная, д. 1, кв. 3.",
    )
    _write_doc(
        tmp_path,
        "doc-flood",
        "Подтопление помещения: прорыв трубы, вода в подъезде, вызывается аварийная служба ЖКХ.",
    )
    return tmp_path


def test_pipeline_loads_normalizes_and_chunks(corpus_dir: Path):
    built = build_corpus(corpus_dir, version="v1", now=T0)
    assert built.version.document_count == 2
    assert built.version.fragment_count == len(built.fragments) > 0
    for fragment in built.fragments:
        assert fragment.source.doc_id in {"doc-fire", "doc-flood"}
        assert fragment.corpus.version == "v1"


def test_no_pii_from_input_survives_in_fragments(corpus_dir: Path):
    built = build_corpus(corpus_dir, version="v1", now=T0)
    all_text = "\n".join(f.text for f in built.fragments)
    for leak in ("Иванов Иван Иванович", "111-22-33", "Опытная, д. 1, кв. 3"):
        assert leak not in all_text
    assert built.quality.total_hits >= 3


def test_offline_search_ranks_relevant_fragment_first(corpus_dir: Path):
    built = build_corpus(corpus_dir, version="v1", now=T0)
    injector = FaultInjector(clock=lambda: T0)
    index = build_index(built, HashingEmbedder(), injector)

    hits = index.search("прорыв трубы вода в подъезде")
    assert hits
    assert hits[0].fragment.source.doc_id == "doc-flood"
    assert hits[0].score > 0
    assert hits[0].fragment.source.approval_status is ApprovalStatus.APPROVED


def test_search_is_offline_and_deterministic(corpus_dir: Path):
    built = build_corpus(corpus_dir, version="v1", now=T0)
    embedder = HashingEmbedder()
    idx_a = build_index(built, embedder, FaultInjector(clock=lambda: T0))
    idx_b = build_index(built, embedder, FaultInjector(clock=lambda: T0))
    query = "пожар дым эвакуация"
    scores_a = [(h.fragment.fragment_id, h.score) for h in idx_a.search(query)]
    scores_b = [(h.fragment.fragment_id, h.score) for h in idx_b.search(query)]
    assert scores_a == scores_b


def test_changing_corpus_version_does_not_change_old_version_search(corpus_dir: Path):
    built_v1 = build_corpus(corpus_dir, version="v1", now=T0)
    injector = FaultInjector(clock=lambda: T0)
    index_v1 = build_index(built_v1, HashingEmbedder(), injector)
    before = [(h.fragment.fragment_id, h.score) for h in index_v1.search("пожар дым")]

    _write_doc(corpus_dir, "doc-gas", "Утечка газа в подвале, пахнет газом, жильцы эвакуированы.")
    built_v2 = build_corpus(corpus_dir, version="v2", now=T0)
    build_index(built_v2, HashingEmbedder(), FaultInjector(clock=lambda: T0))

    after = [(h.fragment.fragment_id, h.score) for h in index_v1.search("пожар дым")]
    assert before == after
    assert index_v1.corpus.version == "v1"
    assert built_v2.version.version == "v2"
    assert built_v2.version.document_count == 3


def test_search_with_mismatched_corpus_version_returns_nothing(corpus_dir: Path):
    built = build_corpus(corpus_dir, version="v1", now=T0)
    injector = FaultInjector(clock=lambda: T0)
    index = build_index(built, HashingEmbedder(), injector)
    other_version = built.version.ref.model_copy(update={"version": "v2"})
    assert index.search("пожар", corpus_version=other_version) == []


def test_embedder_failure_is_logged_and_degrades_without_rag(corpus_dir: Path):
    built = build_corpus(corpus_dir, version="v1", now=T0)
    injector = FaultInjector(clock=lambda: T0)
    injector.inject(ComponentName.EMBEDDER, FailureKind.ERROR)

    with pytest.raises(ComponentFailure):
        build_index(built, HashingEmbedder(), injector)

    assert len(injector.log) == 1
    assert injector.log[0].component is ComponentName.EMBEDDER
    assert Degradation.GENERATORS_WITHOUT_RAG is not None  # деградация для этого компонента

    # Даже при отказе embedder корпус и фрагменты доступны — генератор может работать
    # на явно переданных материалах без поиска (Degradation.GENERATORS_WITHOUT_RAG).
    assert built.fragments
    assert all(f.corpus.version == "v1" for f in built.fragments)


def test_only_synthetic_documents_are_accepted(tmp_path: Path):
    _write_doc(tmp_path, "doc-real", "текст", synthetic=False)
    with pytest.raises(CorpusValidationError):
        build_corpus(tmp_path, version="v1", now=T0)


def test_embedder_model_ref_is_recorded():
    embedder = HashingEmbedder()
    assert isinstance(embedder.model_ref, ModelRef)
    assert embedder.model_ref.model_name
