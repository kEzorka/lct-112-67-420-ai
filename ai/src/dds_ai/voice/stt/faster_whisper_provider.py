"""Адаптер STT на faster-whisper (CTranslate2, CPU).

Extras: `pip install "dds-ai[voice-fasterwhisper]"`.

Не запускался в этом окружении: веса моделей раздаются с huggingface.co, хост заблокирован
сетевой политикой контейнера (см. docs/ai/bench/voice-cpu.md, раздел «Что не проверено»).
Эта обёртка не даёт нативных частичных (partial) результатов — faster-whisper распознаёт
целиком переданный буфер; потоковые partial для этой модели потребовали бы внешней
VAD-нарезки на фронтенде inference-worker (M1/M3), здесь не реализовано.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence

from ...contracts.common import ModelRef
from ...ports import TranscriptSegment


class FasterWhisperSTT:
    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        model_version: str | None = None,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - зависит от опциональной зависимости
            raise ImportError(
                'faster-whisper не установлен: pip install "dds-ai[voice-fasterwhisper]"'
            ) from exc
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_ref = ModelRef(
            component="stt", model_name="faster-whisper", model_version=model_version or model_size
        )

    def transcribe(self, audio: bytes, *, hints: Sequence[str] = ()) -> list[TranscriptSegment]:
        initial_prompt = ", ".join(hints) if hints else None
        segments, _info = self._model.transcribe(
            io.BytesIO(audio),
            language="ru",
            initial_prompt=initial_prompt,
            word_timestamps=True,
        )
        out: list[TranscriptSegment] = []
        for seg in segments:
            # avg_logprob — логарифм средней вероятности токена, не откалиброванный confidence;
            # переводим в (0, 1] как ориентир неуверенности, не как вероятность правильности.
            confidence = math.exp(seg.avg_logprob) if seg.avg_logprob is not None else None
            out.append(
                TranscriptSegment(
                    text=seg.text.strip(),
                    start_ms=int(seg.start * 1000),
                    end_ms=int(seg.end * 1000),
                    final=True,
                    confidence=confidence,
                )
            )
        return out
