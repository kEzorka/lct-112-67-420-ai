#!/usr/bin/env python3
"""Бенчмарк STT-кандидатов на CPU (6.2).

Запуск: `python ai/bench/run_stt_bench.py --audio-dir <dir>`.

Это НЕ целевое железо заказчика (раздел 9 промпта) — фиксируй процессор отдельно.
Аудио для датасета (`dataset.py`) в репозиторий не входит: сгенерируй его локально
(`synthesize_dataset_audio.py`) или запиши вручную. Аудио, полученное синтезом TTS,
вносит смещение в замер STT (совпадение доменов синтеза и распознавания завышает
качество) — это отмечено отдельно в выводе и обязательно в отчёте.
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import time
from collections.abc import Callable
from pathlib import Path

from dataset import DATASET
from metrics import cer, wer

CANDIDATES: dict[str, Callable[[], object]] = {}

_MODELS_DIR = os.environ.get("DDS_BENCH_MODELS_DIR", "models")


def _register(name: str, factory: Callable[[], object]) -> None:
    CANDIDATES[name] = factory


def _register_known_candidates() -> None:
    try:
        from dds_ai.voice.stt.faster_whisper_provider import FasterWhisperSTT

        _register("faster-whisper-small", lambda: FasterWhisperSTT("small"))
        _register("faster-whisper-medium", lambda: FasterWhisperSTT("medium"))
    except ImportError:
        pass
    try:
        from dds_ai.voice.stt.vosk_provider import VoskSTT

        _register(
            "vosk-ru",
            lambda: VoskSTT(model_path=f"{_MODELS_DIR}/vosk-model-small-ru-0.22"),
        )
    except ImportError:
        pass


def _peak_rss_mb() -> float:
    """Пиковая резидентная память процесса (Linux: ru_maxrss в КиБ)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(candidate_names: list[str], audio_dir: Path | None, *, synthetic_audio: bool) -> None:
    _register_known_candidates()
    print(
        f"host: {platform.platform()}, cpu: {platform.processor() or 'неизвестно'}, "
        f"cores: {os.cpu_count()}"
    )
    print("ВНИМАНИЕ: это не целевое железо заказчика (раздел 9 промпта).")
    if synthetic_audio:
        print(
            "ВНИМАНИЕ: аудио синтезировано через TTS, а не записано живым голосом — "
            "WER здесь смещён в лучшую сторону относительно реальной телефонии."
        )

    for name in candidate_names:
        factory = CANDIDATES.get(name)
        if factory is None:
            print(f"[{name}] недоступен: неизвестный кандидат или нет опциональной зависимости")
            continue
        try:
            provider = factory()
        except Exception as exc:  # нет пакета, нет весов, не удалось загрузить модель
            print(f"[{name}] недоступен: {exc}")
            continue

        if audio_dir is None:
            print(f"[{name}] загружен, но пропущен: не передан --audio-dir")
            continue

        total_wer = total_cer = total_latency_s = 0.0
        n = 0
        # По категориям сущностей (адрес/число/отрицание/служба) — раздел 6.2 промпта:
        # общий WER/CER скрывает, на какой категории модель ошибается больше всего.
        category_flags = {
            "address": "has_address",
            "number": "has_number",
            "negation": "has_negation",
            "service": "has_service",
        }
        category_wer: dict[str, list[float]] = {c: [] for c in category_flags}
        category_cer: dict[str, list[float]] = {c: [] for c in category_flags}
        for utt in DATASET:
            audio_path = audio_dir / f"{utt.id}.wav"
            if not audio_path.exists():
                print(f"[{name}] нет аудио для {utt.id}: {audio_path}")
                continue
            audio = audio_path.read_bytes()
            start = time.monotonic()
            segments = provider.transcribe(audio)
            latency_s = time.monotonic() - start
            hypothesis = " ".join(s.text for s in segments if s.final)
            utt_wer = wer(utt.text, hypothesis)
            utt_cer = cer(utt.text, hypothesis)
            total_wer += utt_wer
            total_cer += utt_cer
            total_latency_s += latency_s
            n += 1
            for category, flag in category_flags.items():
                if getattr(utt, flag):
                    category_wer[category].append(utt_wer)
                    category_cer[category].append(utt_cer)

        if n == 0:
            print(f"[{name}] нет доступного аудио — замер не выполнен")
            continue
        by_category = " ".join(
            f"{cat}: WER={sum(category_wer[cat]) / len(category_wer[cat]):.3f} "
            f"CER={sum(category_cer[cat]) / len(category_cer[cat]):.3f} "
            f"(n={len(category_wer[cat])})"
            for cat in category_flags
            if category_wer[cat]
        )
        print(
            f"[{name}] model_ref={provider.model_ref} n={n} "
            f"WER={total_wer / n:.3f} CER={total_cer / n:.3f} "
            f"avg_final_latency_s={total_latency_s / n:.3f} peak_rss_mb={_peak_rss_mb():.1f}"
        )
        print(f"[{name}] по сущностям: {by_category}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["faster-whisper-small", "vosk-ru"],
        help="Имена кандидатов (см. CANDIDATES в этом файле)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Каталог с <utterance_id>.wav для DATASET; без него замер не выполняется",
    )
    parser.add_argument(
        "--synthetic-audio",
        action="store_true",
        help="Пометить, что аудио получено синтезом TTS (смещение WER)",
    )
    args = parser.parse_args()
    run(args.candidates, args.audio_dir, synthetic_audio=args.synthetic_audio)
