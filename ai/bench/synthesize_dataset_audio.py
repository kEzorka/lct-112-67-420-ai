#!/usr/bin/env python3
"""Генерирует синтетическое аудио датасета через TTS-кандидат — вход для run_stt_bench.py.

Аудио не коммитится (`*.wav` в `.gitignore`). Синтетическая речь как вход STT — смещённый
замер (совпадение доменов синтеза/распознавания завышает качество): обязательно передавай
`--synthetic-audio` в `run_stt_bench.py` и отмечай это в отчёте.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

from dataset import DATASET
from run_tts_bench import CANDIDATES, _register_known_candidates


def main() -> None:
    _register_known_candidates()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tts-candidate", required=True, choices=sorted(CANDIDATES) or ["<нет доступных>"]
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="По умолчанию берётся из provider.sample_rate (частота, на которой обучена модель)",
    )
    args = parser.parse_args()

    entry = CANDIDATES.get(args.tts_candidate)
    if entry is None:
        raise SystemExit(f"кандидат {args.tts_candidate!r} недоступен (нет весов или зависимости)")
    factory, voice_id = entry
    provider = factory()
    sample_rate = args.sample_rate or getattr(provider, "sample_rate", 22050)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for utt in DATASET:
        pcm = provider.synthesize(utt.text, voice=voice_id, intonation="neutral")
        path = args.out_dir / f"{utt.id}.wav"
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
    print(f"записано {len(DATASET)} файлов в {args.out_dir} (синтетическая речь, не живой голос)")


if __name__ == "__main__":
    main()
