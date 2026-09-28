#!/usr/bin/env python3
"""Прогон контрольной выборки калибровки (D-039) через `LLMSemanticJudge`.

Использование:
    python ai/calibration/run_calibration.py [--fake] [--model PATH] [--json OUT.json]

По умолчанию ищет один файл `*.gguf` в `<repo_root>/models/` (веса не коммитятся, см.
`.gitignore`) и запускает настоящую модель через `LlamaCppProvider` в свободном режиме
(`json_grammar=False`) — решётка (grammar) адаптера жёстко привязана к схеме реплики
руководителя (`{"action","text"}`), поэтому для другой схемы оценивателя её нужно отключать
(см. `docs/ai/bench/llm-cpu.md`, наблюдение замера). Модели нет — фолбэк на детерминированный
скриптовый ответчик (`--fake` включает его принудительно); отчёт обязан честно называть, какой
случай был на самом деле (D-039: «без разметки от преподавателя — не заявлять точность»).

Метрики — по каждому критерию и по всей выборке: согласие с рабочей меткой (не эталонной
разметкой преподавателя), ложные штрафы (`full`/`partial`→`none`), пропуски ошибок
(`none`→`full`/`partial`), доля `not_checked`/`unclear`, сравнение с порогами W-03 без
заявления о достижении.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import CONVERSATION_CASES, FIELD_CASES, ExpectedLabel  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "ai" / "src"))

from dds_ai.contracts.common import ModelRef  # noqa: E402
from dds_ai.faults import InvalidOutput  # noqa: E402
from dds_ai.judging.policy import JudgePolicy  # noqa: E402
from dds_ai.judging.schema import MatchLevel  # noqa: E402
from dds_ai.judging.semantic_judge import LLMSemanticJudge  # noqa: E402

# Калибровка измеряет качество классификации модели (full/partial/none/unclear), а не
# приложенческую политику F-2 (`JudgePolicy`, по умолчанию пустая — модель одна не выносит
# `failed`). Иначе `none` и `unclear` слились бы в один бакет `not_checked`, и отчёт перестал
# бы показывать, распознаёт ли модель `none` вообще (см. «Главное наблюдение» в отчёте).
CALIBRATION_POLICY = JudgePolicy(
    failable_criteria=frozenset({"card.circumstances", "manual.additions"})
)

W03_THRESHOLDS = {
    "agreement": 0.85,
    "false_penalty": 0.05,
    "missed_error": 0.10,
    "not_checked_share": 0.10,
}

# not_checked (unclear/invalid) не в счёт как ошибка калибровки: unclear на criterion уровне
# уходит в CriterionStatus.NOT_CHECKED, это ожидаемое поведение при повреждённом транскрипте.
_MATCH_TO_LABEL = {
    MatchLevel.FULL: ExpectedLabel.FULL,
    MatchLevel.PARTIAL: ExpectedLabel.PARTIAL,
    MatchLevel.NONE: ExpectedLabel.NONE,
    MatchLevel.UNCLEAR: ExpectedLabel.UNCLEAR,
}

_POSITIVE = {ExpectedLabel.FULL, ExpectedLabel.PARTIAL}
_NEGATIVE = {ExpectedLabel.NONE}


class ScriptedFallbackLLM:
    """Фолбэк, когда весов нет: не имитирует качество модели, только держит форму выхода
    рабочей, чтобы обвязку калибровки можно было проверить без сети (честно помечается)."""

    model_ref = ModelRef(component="llm", model_name="scripted-fallback", model_version="0")

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        import re

        if "<текст_диспетчера>" in prompt:
            m = re.search(r"<текст_диспетчера>\n(.*?)\n</текст_диспетчера>", prompt, re.S)
            text = (m.group(1) if m else "").strip()
            quote = text.split(".")[0][:30].strip()
            data = {
                "match": "full" if text else "unclear",
                "quote": quote,
                "reason_code": "exact_match",
            }
            return json.dumps(data, ensure_ascii=False)
        m_items = re.search(r"<элементы_доклада>\n(.*?)\n</элементы_доклада>", prompt, re.S)
        m_tr = re.search(r"<транскрипт_диспетчера>\n(.*?)\n</транскрипт_диспетчера>", prompt, re.S)
        items_text = (m_items.group(1) if m_items else "").strip()
        transcript = (m_tr.group(1) if m_tr else "").strip()
        ids = [line.split(":", 1)[0].strip() for line in items_text.splitlines() if line.strip()]
        quote = transcript.split(".")[0][:30].strip()
        return json.dumps(
            {"items": [{"item_id": i, "match": "full", "quote": quote} for i in ids]},
            ensure_ascii=False,
        )


def find_model() -> Path | None:
    models_dir = REPO_ROOT / "models"
    if not models_dir.is_dir():
        return None
    candidates = sorted(models_dir.glob("*.gguf"))
    return candidates[0] if candidates else None


def build_judge(
    *,
    force_fake: bool,
    model_path: Path | None,
    timeout_s: float = 30,
    n_threads: int | None = None,
) -> tuple[LLMSemanticJudge, bool]:
    if not force_fake:
        path = model_path or find_model()
        if path is not None:
            try:
                from dds_ai.llm.llamacpp_provider import LlamaCppProvider

                llm = LlamaCppProvider(
                    path, timeout_s=timeout_s, n_threads=n_threads, json_grammar=False
                )
                return LLMSemanticJudge(llm, policy=CALIBRATION_POLICY), True
            except Exception as exc:  # pragma: no cover - зависит от окружения
                msg = f"# не удалось загрузить {path}: {exc}; фолбэк на скриптовый ответ"
                print(msg, file=sys.stderr)
    return LLMSemanticJudge(ScriptedFallbackLLM(), policy=CALIBRATION_POLICY), False


def run_case(judge: LLMSemanticJudge, case) -> dict:
    started = time.monotonic()
    try:
        result = judge.judge(case.criterion_id, case.context)
        predicted = _from_result(result)
        error = None
    except InvalidOutput as exc:
        predicted = ExpectedLabel.UNCLEAR  # невалидный выход -> not_checked, тот же бакет
        error = f"invalid_output: {exc}"
    latency = time.monotonic() - started
    return {
        "case_id": case.case_id,
        "source_ref": case.source_ref,
        "category": case.category.value,
        "criterion_id": case.criterion_id,
        "expected": case.expected_label.value,
        "predicted": predicted.value,
        "latency_s": round(latency, 2),
        "error": error,
    }


def _from_result(result) -> ExpectedLabel:
    from dds_ai.contracts.criteria import CriterionStatus

    if result.status is CriterionStatus.NOT_CHECKED:
        return ExpectedLabel.UNCLEAR
    if result.status is CriterionStatus.PASSED and result.value == 1:
        return ExpectedLabel.FULL
    if result.status is CriterionStatus.PASSED and result.value == 0.5:
        return ExpectedLabel.PARTIAL
    if result.status is CriterionStatus.FAILED:
        return ExpectedLabel.NONE
    return ExpectedLabel.UNCLEAR


def compute_metrics(rows: list[dict]) -> dict:
    by_criterion: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_criterion[r["criterion_id"]].append(r)

    def metrics_for(subset: list[dict]) -> dict:
        n = len(subset)
        agree = sum(1 for r in subset if r["predicted"] == r["expected"])
        false_penalty = sum(
            1 for r in subset if r["expected"] in _POSITIVE_S and r["predicted"] in _NEGATIVE_S
        )
        missed_error = sum(
            1 for r in subset if r["expected"] in _NEGATIVE_S and r["predicted"] in _POSITIVE_S
        )
        not_checked = sum(1 for r in subset if r["predicted"] == ExpectedLabel.UNCLEAR.value)
        eligible_penalty = sum(1 for r in subset if r["expected"] in _POSITIVE_S)
        eligible_missed = sum(1 for r in subset if r["expected"] in _NEGATIVE_S)
        return {
            "n": n,
            "agreement": round(agree / n, 3) if n else None,
            "false_penalty_rate": round(false_penalty / eligible_penalty, 3)
            if eligible_penalty
            else None,
            "missed_error_rate": round(missed_error / eligible_missed, 3)
            if eligible_missed
            else None,
            "not_checked_share": round(not_checked / n, 3) if n else None,
        }

    return {
        "overall": metrics_for(rows),
        "by_criterion": {cid: metrics_for(rs) for cid, rs in sorted(by_criterion.items())},
        "latency_s": _latency_stats(rows),
    }


def _latency_stats(rows: list[dict]) -> dict:
    values = sorted(r["latency_s"] for r in rows)
    n = len(values)
    if n == 0:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None}

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return values[idx]

    return {
        "n": n,
        "min": round(values[0], 2),
        "median": round(pct(0.5), 2),
        "p95": round(pct(0.95), 2),
        "max": round(values[-1], 2),
    }


_POSITIVE_S = {v.value for v in _POSITIVE}
_NEGATIVE_S = {v.value for v in _NEGATIVE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="принудительно скриптовый ответчик")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--timeout-s", type=float, default=30, help="срок генерации одного вызова")
    ap.add_argument("--n-threads", type=int, default=None, help="потоки CPU (по умолчанию авто)")
    args = ap.parse_args()

    judge, real_model = build_judge(
        force_fake=args.fake,
        model_path=args.model,
        timeout_s=args.timeout_s,
        n_threads=args.n_threads,
    )
    cases = list(FIELD_CASES) + list(CONVERSATION_CASES)
    rows = [run_case(judge, c) for c in cases]
    metrics = compute_metrics(rows)

    out = {
        "real_model": real_model,
        "model_ref": judge.model_ref.model_dump(mode="json"),
        "prompt_version": judge.prompt.version,
        "sample_size": len(rows),
        "source_refs": sorted({r["source_ref"] for r in rows}),
        "thresholds_w03": W03_THRESHOLDS,
        "metrics": metrics,
        "rows": rows,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
