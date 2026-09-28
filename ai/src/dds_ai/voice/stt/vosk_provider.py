"""Адаптер STT на Vosk (Kaldi, CPU, потоковый). Extras: `pip install "dds-ai[voice-vosk]"`.

Не запускался в этом окружении: модели раздаются с alphacephei.com, хост заблокирован
сетевой политикой контейнера (см. docs/ai/bench/voice-cpu.md). Vosk отдаёt пословный
confidence при `SetWords(True)` — сегмент получает confidence как среднее по словам.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ...contracts.common import ModelRef
from ...ports import TranscriptSegment


class VoskSTT:
    def __init__(
        self, model_path: str, *, sample_rate: int = 16000, model_version: str | None = None
    ):
        try:
            import vosk
        except ImportError as exc:  # pragma: no cover - зависит от опциональной зависимости
            raise ImportError('vosk не установлен: pip install "dds-ai[voice-vosk]"') from exc
        self._vosk = vosk
        self._sample_rate = sample_rate
        self._model = vosk.Model(model_path)
        self.model_ref = ModelRef(
            component="stt", model_name="vosk", model_version=model_version or model_path
        )

    def transcribe(self, audio: bytes, *, hints: Sequence[str] = ()) -> list[TranscriptSegment]:
        grammar = json.dumps(list(hints), ensure_ascii=False) if hints else None
        recognizer = (
            self._vosk.KaldiRecognizer(self._model, self._sample_rate, grammar)
            if grammar
            else self._vosk.KaldiRecognizer(self._model, self._sample_rate)
        )
        recognizer.SetWords(True)
        recognizer.AcceptWaveform(audio)
        result = json.loads(recognizer.FinalResult())

        words = result.get("result", [])
        text = result.get("text", "")
        if not text:
            return []
        confidences = [w["conf"] for w in words if "conf" in w]
        confidence = sum(confidences) / len(confidences) if confidences else None
        start_ms = int(words[0]["start"] * 1000) if words else 0
        end_ms = int(words[-1]["end"] * 1000) if words else 0
        return [
            TranscriptSegment(
                text=text, start_ms=start_ms, end_ms=end_ms, final=True, confidence=confidence
            )
        ]
