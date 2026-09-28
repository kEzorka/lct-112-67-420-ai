#!/usr/bin/env python3
"""Раздельный замер стадий LLM-формулировки реплик руководителя на CPU (M3, раздел 9).

Стадии каждого хода руководителя:
- `generation` — вызов `LLMProvider.complete()` (одна попытка);
- `validation` — разбор и проверка выхода (`supervisor/validator.py`, одна попытка);
- `phrase_total` — от начала формулировки до готового текста или отказа (очередь воркера,
  все попытки, повтор после невалидного выхода);
- `turn_total` — весь ход `Conversation.hear()`: автомат, шаблон, формулировка, fallback.

Это НЕ целевое железо и НЕ задержка до первого звука: TTS и транспорт сюда не входят (6.4,
D-052). Кандидаты моделей — предмет замера, не решение (раздел 11).

Провайдеры:
- `echo` — детерминированный фейк (возвращает черновик) с заданной задержкой: проверка
  обвязки, а не модели;
- `llamacpp --model PATH.gguf` — локальный файл GGUF через `LlamaCppProvider`
  (`pip install "dds-ai[llm-llamacpp]"`), ничего не скачивается.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections import Counter

from dds_ai.contracts.dialogue import ReplyMode
from dds_ai.contracts.events import ComponentName
from dds_ai.cycle import load_synthetic
from dds_ai.llm import EchoDraftLLM
from dds_ai.supervisor import Supervisor
from dds_ai.supervisor.phrasing import LLMPhraser
from dds_ai.worker import InferenceWorker

# Синтетические реплики диспетчера по сценариям (не билеты, не эталон).
SCRIPTS: dict[str, tuple[str, ...]] = {
    "syn-001-fire-respond": (
        "Сколько пострадавших?",
        "Пожар в жилом доме, ул. Тестовая, д. 12, кв. 5.",
        "Дым из окна квартиры. ДДС реагирует. Прошу принять доклад.",
        "Да, верно.",
    ),
    "syn-002-gas-redirect": (
        "Есть пострадавшие?",
        "Запах газа в подъезде, ул. Условная, д. 7. Перенаправляем в газовую службу. "
        "Прошу принять к сведению.",
        "Да.",
    ),
    "syn-003-repeat-refuse": (
        "Повторное сообщение о мусоре, ул. Модельная, д. 3.",
        "Возгорание уже потушено, ДДС отказывает в реагировании. Прошу принять для сведения.",
        "Да, верно.",
    ),
}


def make_llm(args: argparse.Namespace):
    if args.provider == "echo":
        return EchoDraftLLM(latency_s=args.fake_latency_ms / 1000)
    from dds_ai.llm.llamacpp_provider import LlamaCppProvider

    return LlamaCppProvider(
        args.model,
        model_version=args.model_version,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        timeout_s=args.timeout_s,
    )


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ms = [v * 1000 for v in values]
    return {
        "n": len(ms),
        "p50_ms": round(statistics.median(ms), 2),
        "p95_ms": round(pct(ms, 0.95), 2),
        "max_ms": round(max(ms), 2),
    }


def run(args: argparse.Namespace) -> dict:
    llm = make_llm(args)
    worker = InferenceWorker.from_profile("cpu")
    worker.register(ComponentName.LLM, llm)
    phraser = LLMPhraser(llm, worker, role_title="руководитель дежурной смены")
    scenarios = load_synthetic()

    gen, val, phrase, turn = [], [], [], []
    modes: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    for _ in range(args.runs):
        for name, script in SCRIPTS.items():
            s = scenarios[name]
            sv = Supervisor(s.brief("supervisor"), scenario=s.scenario, phraser=phraser)
            result = sv.dial()
            while not result.connected:  # сценарные «занято» / «не отвечает»
                result = sv.dial()
            conv = result.conversation
            turns = [result.greeting]
            for text in script:
                if conv.phase == "ended":
                    break
                started = time.monotonic()
                t = conv.hear(text)
                turn.append(time.monotonic() - started)
                turns.append(t)
            for t in turns:
                if t is None or t.reply is None:
                    continue
                modes[t.reply.mode.value] += 1
                if t.llm is None:
                    continue
                phrase.append(t.llm.total_s)
                for a in t.llm.attempts:
                    outcomes[a.outcome] += 1
                    if a.generation_s is not None:
                        gen.append(a.generation_s)
                    if a.validation_s is not None:
                        val.append(a.validation_s)
    worker.shutdown()
    return {
        "note": "не целевое железо; без TTS и транспорта; не решение о модели",
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "provider": args.provider,
        "model_ref": llm.model_ref.model_dump(),
        "prompt_version": phraser.template.version,
        "fake_latency_ms": args.fake_latency_ms if args.provider == "echo" else None,
        "runs": args.runs,
        "concurrency": 1,
        "stages": {
            "generation": summary(gen),
            "validation": summary(val),
            "phrase_total": summary(phrase),
            "turn_total": summary(turn),
        },
        "reply_modes": dict(modes),
        "attempt_outcomes": dict(outcomes),
        "fallback_share": round(modes[ReplyMode.FALLBACK.value] / max(sum(modes.values()), 1), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Замер стадий LLM-формулировки (не целевое железо)")
    ap.add_argument("--provider", choices=("echo", "llamacpp"), default="echo")
    ap.add_argument("--model", help="путь к локальному файлу GGUF (для llamacpp)")
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--fake-latency-ms", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()
    if args.provider == "llamacpp" and not args.model:
        ap.error("--model is required for llamacpp")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
