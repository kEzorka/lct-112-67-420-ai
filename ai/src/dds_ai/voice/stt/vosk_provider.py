"""Адаптер STT на Vosk (Kaldi, CPU, потоковый). Extras: `pip install "dds-ai[voice-vosk]"`.

Реально прогонялся в бенчмарке (`docs/ai/bench/voice-cpu.md`) на модели
`vosk-model-small-ru-0.22` с alphacephei.com. Vosk отдаёт пословный confidence при
`SetWords(True)` — сегмент получает confidence как среднее по словам.

`AcceptWaveform` ожидает «сырой» PCM s16 mono на частоте, с которой создан распознаватель —
в отличие от `FasterWhisperSTT.transcribe`, который принимает целый контейнер (WAV/etc.) и
сам его декодирует через ffmpeg. Чтобы оба адаптера принимали один и тот же вход
`audio: bytes` в бенчмарке (`run_stt_bench.py` читает `<utterance>.wav` как есть), этот
адаптер сам распознаёт WAV-контейнер (`RIFF`-заголовок) и достаёт из него PCM-кадры и
фактическую частоту дискретизации; если заголовка нет — считает `audio` уже сырым PCM на
частоте `sample_rate` из конструктора.
"""

from __future__ import annotations

import io
import json
import wave
from collections.abc import Sequence

from ...contracts.common import ModelRef
from ...ports import TranscriptSegment


def _pcm_and_rate(audio: bytes, default_sample_rate: int) -> tuple[bytes, int]:
    if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes()), wav_file.getframerate()
    return audio, default_sample_rate


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
        pcm, sample_rate = _pcm_and_rate(audio, self._sample_rate)
        grammar = json.dumps(list(hints), ensure_ascii=False) if hints else None
        recognizer = (
            self._vosk.KaldiRecognizer(self._model, sample_rate, grammar)
            if grammar
            else self._vosk.KaldiRecognizer(self._model, sample_rate)
        )
        recognizer.SetWords(True)
        recognizer.AcceptWaveform(pcm)
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
