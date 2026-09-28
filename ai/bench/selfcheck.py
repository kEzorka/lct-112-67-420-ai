#!/usr/bin/env python3
"""Самопроверка обвязки бенчмарка на фейковых провайдерах — НЕ измерение качества модели.

Подтверждает, что датасет, метрики WER/CER и цикл замера работают до того, как станут
доступны веса реальных кандидатов (см. docs/ai/bench/voice-cpu.md, «Что не проверено»).
"""

from __future__ import annotations

from dataset import DATASET
from metrics import cer, wer

from dds_ai.ports import TranscriptSegment
from dds_ai.voice.stt.fake import FakeSTTProvider
from dds_ai.voice.tts.fake import FakeTTSProvider


def main() -> None:
    tts = FakeTTSProvider()
    for utt in DATASET:
        audio = tts.synthesize(utt.text, voice="fake", intonation="neutral")
        stt = FakeSTTProvider(
            [TranscriptSegment(text=utt.text, start_ms=0, end_ms=1000, final=True)]
        )
        segments = stt.transcribe(audio)
        hypothesis = " ".join(s.text for s in segments if s.final)
        assert wer(utt.text, hypothesis) == 0.0
        assert cer(utt.text, hypothesis) == 0.0
    print(f"обвязка бенчмарка исправна: {len(DATASET)} реплик, WER/CER=0 на фейковых провайдерах")


if __name__ == "__main__":
    main()
