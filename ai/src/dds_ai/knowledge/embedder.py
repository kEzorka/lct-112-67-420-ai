"""Базовый офлайн embedder без весов (6.10): хеширующий bag-of-words.

Реализует порт `ports.Embedder`. Детерминирован, не требует сети, файлов моделей
или обучения — годится как встроенный fallback; настоящая embedding-модель
подключается опционально через тот же порт, код индекса не меняется.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from ..contracts.common import ModelRef

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_index(token: str, dims: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dims


class HashingEmbedder:
    """Хеширующий TF-подобный embedder: одинаковый текст → одинаковый вектор всегда."""

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.model_ref = ModelRef(component="embedder", model_name="hashing-bow", model_version="1")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        tokens = _tokenize(text)
        for token in tokens:
            vec[_hash_index(token, self.dims)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec
