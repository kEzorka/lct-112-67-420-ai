"""Загрузка и валидация исходных документов корпуса (6.10: загрузка → валидация).

Путь к корпусу — конфигурация вызывающего (`source_dir`); код не хранит путь к
синтетическому набору как константу, чтобы настоящий корпус подключался без правок.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..contracts.knowledge import SourceDocument


class CorpusValidationError(Exception):
    pass


@dataclass(frozen=True)
class RawDocument:
    meta: SourceDocument
    text: str


def load_documents(source_dir: Path) -> list[RawDocument]:
    docs = [_load_one(p) for p in sorted(source_dir.glob("*.json"))]
    if not docs:
        raise CorpusValidationError(f"{source_dir}: no *.json documents found")
    _validate_corpus(docs)
    return docs


def _load_one(path: Path) -> RawDocument:
    raw = json.loads(path.read_text("utf-8"))
    text = raw.pop("text", None)
    if not text or not text.strip():
        raise CorpusValidationError(f"{path}: empty text")
    try:
        meta = SourceDocument.model_validate(raw)
    except Exception as exc:
        raise CorpusValidationError(f"{path}: {exc}") from exc
    return RawDocument(meta=meta, text=text)


def _validate_corpus(docs: list[RawDocument]) -> None:
    seen: set[str] = set()
    for d in docs:
        if not d.meta.synthetic:
            raise CorpusValidationError(
                f"{d.meta.doc_id}: только явно синтетические документы — исходники "
                "заказчика в репозиторий не коммитятся"
            )
        if d.meta.doc_id in seen:
            raise CorpusValidationError(f"duplicate doc_id: {d.meta.doc_id}")
        seen.add(d.meta.doc_id)
