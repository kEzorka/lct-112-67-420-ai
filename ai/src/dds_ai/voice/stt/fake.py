"""Детерминированный STT без весов модели — для тестов и проводки пайплайна (6.2)."""

from __future__ import annotations

from collections.abc import Sequence

from ...contracts.common import ModelRef
from ...ports import TranscriptSegment


class FakeSTTProvider:
    model_ref = ModelRef(component="stt", model_name="fake", model_version="0")

    def __init__(self, segments: Sequence[TranscriptSegment] = ()):
        self._segments = tuple(segments)
        self.calls: list[bytes] = []

    def transcribe(self, audio: bytes, *, hints: Sequence[str] = ()) -> list[TranscriptSegment]:
        self.calls.append(audio)
        return list(self._segments)
