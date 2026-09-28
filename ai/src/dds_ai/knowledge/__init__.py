"""База знаний / RAG (6.10): офлайн pipeline и поиск по версии корпуса.

Заменяемый модуль команды (не требование ТЗ, Miro 07): генераторы 6.5 продолжают
работать на явно переданных материалах, если embedder или индекс недоступны.
"""

from .anonymize import AnonymizeHit, anonymize_text
from .chunking import chunk_text
from .documents import CorpusValidationError, RawDocument, load_documents
from .embedder import HashingEmbedder
from .index import KnowledgeIndex, SearchHit
from .normalize import normalize_text
from .pipeline import BuiltCorpus, QualityReport, build_corpus, build_index

__all__ = [
    "AnonymizeHit",
    "BuiltCorpus",
    "CorpusValidationError",
    "HashingEmbedder",
    "KnowledgeIndex",
    "QualityReport",
    "RawDocument",
    "SearchHit",
    "anonymize_text",
    "build_corpus",
    "build_index",
    "chunk_text",
    "load_documents",
    "normalize_text",
]
