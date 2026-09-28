"""Чанкинг нормализованного текста документа на фрагменты корпуса (6.10)."""

from __future__ import annotations


def chunk_text(text: str, *, max_chars: int = 500) -> list[str]:
    """Склеить абзацы в чанки не длиннее max_chars, не разрывая абзац посередине."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
