"""Детерминированный TTS без весов модели — для тестов и проводки пайплайна (6.3)."""

from __future__ import annotations

from ...contracts.common import ModelRef


class FakeTTSProvider:
    model_ref = ModelRef(component="tts", model_name="fake", model_version="0")

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def synthesize(self, text: str, *, voice: str, intonation: str) -> bytes:
        self.calls.append((text, voice, intonation))
        return f"WAV:{voice}:{intonation}:{text}".encode()
