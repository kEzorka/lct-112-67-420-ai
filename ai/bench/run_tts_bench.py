#!/usr/bin/env python3
"""Бенчмарк TTS-кандидатов на CPU (6.3): время до готовности аудио и RTF.

Это НЕ целевое железо заказчика (раздел 9 промпта). Не объявляй модель утверждённой по
результатам этого замера — лицензии кандидатов см. в `docs/ai/licenses.md`.

Ограничение: `TTSProvider.synthesize()` не потоковый, поэтому «время до готовности аудио»
здесь — это время до полного буфера, а не до первого звука в потоковом смысле D-025/6.3.
Честная оценка задержки до первого звука в потоковом режиме требует реализации streaming-
API конкретной модели — отмечено как непроверенное в отчёте.
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import time
from collections.abc import Callable

from dataset import DATASET

CANDIDATES: dict[str, tuple[Callable[[], object], str]] = {}

_MODELS_DIR = os.environ.get("DDS_BENCH_MODELS_DIR", "models")


def _register(name: str, factory: Callable[[], object], voice_id: str) -> None:
    CANDIDATES[name] = (factory, voice_id)


def _register_known_candidates() -> None:
    try:
        from dds_ai.voice.tts.piper_provider import PiperTTS

        _register(
            "piper-irina",
            lambda: PiperTTS(f"{_MODELS_DIR}/ru_RU-irina-medium.onnx", voice_name="irina"),
            "irina",
        )
        _register(
            "piper-denis",
            lambda: PiperTTS(f"{_MODELS_DIR}/ru_RU-denis-medium.onnx", voice_name="denis"),
            "denis",
        )
    except ImportError:
        pass
    try:
        from dds_ai.voice.tts.silero_provider import SileroTTS

        _register(
            "silero-baya",
            lambda: SileroTTS(f"{_MODELS_DIR}/v4_ru.pt", speaker="baya"),
            "baya",
        )
        _register(
            "silero-aidar",
            lambda: SileroTTS(f"{_MODELS_DIR}/v4_ru.pt", speaker="aidar"),
            "aidar",
        )
    except ImportError:
        pass


def _estimate_duration_s(pcm: bytes, *, sample_rate: int, sample_width: int = 2) -> float:
    """Оценка длительности PCM s16 mono по частоте дискретизации провайдера."""
    return len(pcm) / (sample_rate * sample_width)


def _peak_rss_mb() -> float:
    """Пиковая резидентная память процесса (Linux: ru_maxrss в КиБ)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(candidate_names: list[str]) -> None:
    _register_known_candidates()
    print(
        f"host: {platform.platform()}, cpu: {platform.processor() or 'неизвестно'}, "
        f"cores: {os.cpu_count()}"
    )
    print("ВНИМАНИЕ: это не целевое железо заказчика (раздел 9 промпта).")

    for name in candidate_names:
        entry = CANDIDATES.get(name)
        if entry is None:
            print(f"[{name}] недоступен: неизвестный кандидат или нет опциональной зависимости")
            continue
        factory, voice_id = entry
        load_start = time.monotonic()
        try:
            provider = factory()
        except Exception as exc:  # нет пакета, нет весов, не удалось загрузить модель
            print(f"[{name}] недоступен: {exc}")
            continue
        load_time_s = time.monotonic() - load_start
        sample_rate = getattr(provider, "sample_rate", 22050)

        for intonation in ("neutral", "tense"):
            total_time_s = total_rtf = 0.0
            n = 0
            for utt in DATASET:
                start = time.monotonic()
                audio = provider.synthesize(utt.text, voice=voice_id, intonation=intonation)
                elapsed_s = time.monotonic() - start
                audio_s = _estimate_duration_s(audio, sample_rate=sample_rate)
                total_time_s += elapsed_s
                total_rtf += (elapsed_s / audio_s) if audio_s else float("nan")
                n += 1
            print(
                f"[{name}/{intonation}] model_ref={provider.model_ref} n={n} "
                f"load_time_s={load_time_s:.3f} avg_time_to_full_audio_s={total_time_s / n:.3f} "
                f"avg_rtf={total_rtf / n:.3f} peak_rss_mb={_peak_rss_mb():.1f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["piper-irina", "piper-denis", "silero-baya", "silero-aidar"],
        help="Имена кандидатов (см. CANDIDATES в этом файле)",
    )
    args = parser.parse_args()
    run(args.candidates)
