"""HTTP-сервис ИИ-контура: контракты и учебная попытка на синтетических сценариях.

Транспорт — черновик до ответа B03 (gRPC или HTTP + WebSocket); схемы сообщений — те же
контракты, что в `/contracts`. Попытки хранятся в памяти процесса: журнал, оценки и профили
хранит бэкенд (docs/ai/agent-prompt.md, 4.3), здесь — только то, что нужно для интеграции и демо.

Модель LLM подключается переменной `DDS_LLM_MODEL_PATH` (локальный GGUF, extra
`ai[llm-llamacpp]`). Без неё руководитель работает на проверенном шаблоне (fallback, C-04), а
семантические критерии получают «не проверено» — итог предварительный.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, TypeAdapter

from dds_ai import __version__
from dds_ai.contracts import CONTRACTS_VERSION, SCHEMA_MODELS
from dds_ai.contracts.events import DispatcherDecision, attempt_event_adapter
from dds_ai.contracts.rubric import Rubric
from dds_ai.cycle import Channel, TrainingSession, load_synthetic
from dds_ai.ports import LLMProvider, SemanticJudge

CONFIG = Path(__file__).resolve().parents[3] / "config"


class WallClock:
    """Реальное время для сервиса; `advance` есть у тестовых часов, здесь он ничего не делает."""

    def __call__(self) -> datetime:
        return datetime.now(UTC)

    def advance(self, seconds: float) -> datetime:
        return self()


class CreateAttempt(BaseModel):
    scenario_id: str
    card_seed: int | None = None


class EditCard(BaseModel):
    values: dict[str, Any] = Field(min_length=1)


class Decide(BaseModel):
    decision: DispatcherDecision


class Say(BaseModel):
    text: str = Field(min_length=1)


class Submit(BaseModel):
    incomplete: bool = False


def _load_models() -> tuple[LLMProvider | None, SemanticJudge | None]:
    path = os.environ.get("DDS_LLM_MODEL_PATH")
    if not path:
        return None, None
    from dds_ai.judging import LLMSemanticJudge
    from dds_ai.llm.llamacpp_provider import LlamaCppProvider

    llm = LlamaCppProvider(path)
    return llm, LLMSemanticJudge(llm)


def _dump(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj


def create_app(*, llm: LLMProvider | None = None, judge: SemanticJudge | None = None) -> FastAPI:
    if llm is None and judge is None:
        llm, judge = _load_models()
    rubric = Rubric.model_validate(json.loads((CONFIG / "rubric.w01.json").read_text("utf-8")))
    scenarios = load_synthetic()
    attempts: dict[UUID, TrainingSession] = {}
    lock = threading.Lock()

    app = FastAPI(
        title="dds-ai",
        version=__version__,
        description="ИИ-контур тренажёра диспетчера ДДС. Контракты — черновик "
        f"{CONTRACTS_VERSION}; сценарии — синтетические.",
    )

    def get(attempt_id: UUID) -> TrainingSession:
        s = attempts.get(attempt_id)
        if s is None:
            raise HTTPException(404, "attempt not found")
        return s

    def events(s: TrainingSession) -> list[Any]:
        return [attempt_event_adapter.dump_python(e, mode="json") for e in s.events]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "contracts_version": CONTRACTS_VERSION,
            "llm": None if llm is None else _dump(llm.model_ref),
        }

    @app.get("/v1/contracts")
    def contracts() -> dict[str, Any]:
        return {"version": CONTRACTS_VERSION, "schemas": sorted(SCHEMA_MODELS)}

    @app.get("/v1/contracts/{name}")
    def contract(name: str) -> dict[str, Any]:
        model = SCHEMA_MODELS.get(name)
        if model is None:
            raise HTTPException(404, "unknown contract")
        return model.json_schema() if isinstance(model, TypeAdapter) else model.model_json_schema()

    @app.get("/v1/scenarios")
    def list_scenarios() -> list[dict[str, Any]]:
        return [
            {"scenario_id": sid, "dds_profile": s.dds_profile, "synthetic": True}
            for sid, s in sorted(scenarios.items())
        ]

    @app.post("/v1/attempts", status_code=201)
    def create_attempt(body: CreateAttempt) -> dict[str, Any]:
        scenario = scenarios.get(body.scenario_id)
        if scenario is None:
            raise HTTPException(404, "unknown scenario")
        s = TrainingSession(
            scenario,
            channel=Channel.TEXT,
            clock=WallClock(),  # type: ignore[arg-type]
            llm=llm,
            judge=judge,
            card_seed=body.card_seed,
        )
        with lock:
            attempts[s.attempt_id] = s
        s.notify()
        return {"attempt_id": str(s.attempt_id), "events": events(s)}

    @app.post("/v1/attempts/{attempt_id}/open")
    def open_card(attempt_id: UUID) -> dict[str, Any]:
        s = get(attempt_id)
        visible = s.open_card()
        return {"card": {k: _dump(v) for k, v in visible.items()}, "events": events(s)}

    @app.post("/v1/attempts/{attempt_id}/card")
    def edit_card(attempt_id: UUID, body: EditCard) -> dict[str, Any]:
        s = get(attempt_id)
        s.edit_card(**body.values)
        return {"events": events(s)}

    @app.post("/v1/attempts/{attempt_id}/decision")
    def decide(attempt_id: UUID, body: Decide) -> dict[str, Any]:
        s = get(attempt_id)
        s.decide(body.decision)
        return {"events": events(s)}

    @app.post("/v1/attempts/{attempt_id}/dial")
    def dial(attempt_id: UUID) -> dict[str, Any]:
        s = get(attempt_id)
        try:
            result = s.dial()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {
            "connected": result.outcome is None,
            "outcome": None if result.outcome is None else str(result.outcome),
            "events": events(s),
        }

    @app.post("/v1/attempts/{attempt_id}/say")
    def say(attempt_id: UUID, body: Say) -> dict[str, Any]:
        s = get(attempt_id)
        try:
            turn = s.say(body.text)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {
            "reply": _dump(turn.reply),
            "ack_id": None if turn.ack_id is None else str(turn.ack_id),
            "events": events(s),
        }

    @app.post("/v1/attempts/{attempt_id}/submit")
    def submit(attempt_id: UUID, body: Submit) -> dict[str, Any]:
        s = get(attempt_id)
        try:
            s.submit(incomplete=body.incomplete)
        except Exception as exc:  # IncompleteSubmission: не хватает шагов
            raise HTTPException(409, str(exc)) from exc
        return {"events": events(s)}

    @app.get("/v1/attempts/{attempt_id}/evaluation")
    def evaluation(attempt_id: UUID) -> dict[str, Any]:
        s = get(attempt_id)
        try:
            ev = s.evaluate(rubric)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {
            "summary": _dump(ev.summary),
            "results": [_dump(r) for r in ev.results],
            "version_snapshot": _dump(s.version_snapshot(rubric)),
        }

    @app.get("/v1/attempts/{attempt_id}/events")
    def attempt_events(attempt_id: UUID) -> list[Any]:
        return events(get(attempt_id))

    return app
