"""Офлайн-прогон (раздел 8 промпта: «полный цикл с отключённой сетью; нет исходящих
соединений»; определение готовности п.3).

Две проверки:
- импорт пакета не открывает сетевых соединений (подпроцесс с заблокированным `socket`,
  до какого-либо импорта `dds_ai` — база пакета зависит только от `pydantic`, реальные модели
  STT/TTS/LLM — опциональные extras, которых нет в обычной установке, см. `ai/pyproject.toml`);
- полный учебный цикл на фейках (без реальных моделей) проходит с заблокированной сетью:
  уведомление → карточка → решение → разговор с руководителем → подтверждение → сдача →
  оценка с объяснением → рекомендация.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dds_ai.contracts.common import VersionRef
from dds_ai.contracts.events import DispatcherDecision
from dds_ai.contracts.profile import PoolTask, ProfileStatus, RecommendationStatus, TaskPool
from dds_ai.contracts.scoring import ScoreVersion, Verdict
from dds_ai.cycle import Channel, TrainingSession, load_synthetic
from dds_ai.mocks.media import MockMediaTransport
from dds_ai.profile import AttemptRecord, build_profile, recommend

from .fakes import FakeLLM, FakeTTS, StubJudge

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
FIRE_REPORT = (
    "Пожар в жилом доме, ул. Тестовая, д. 12, кв. 5. Дым из окна квартиры. "
    "ДДС реагирует, направляем пожарную охрану. Прошу принять доклад."
)


# --- импорт пакета ничего не скачивает -----------------------------------------------------

_IMPORT_SCRIPT = """
import socket

def _blocked(*a, **kw):
    raise AssertionError("network access attempted while importing dds_ai")

socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked

import dds_ai  # noqa: F401
import dds_ai.cycle  # noqa: F401
import dds_ai.worker  # noqa: F401
import dds_ai.voice  # noqa: F401

print("OK")
"""


def test_package_import_opens_no_network_connections():
    """Сеть блокируется до импорта в отдельном процессе — подмена в текущем процессе не
    доказала бы ничего про уже загруженные модули (кеш `sys.modules`)."""
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


# --- полный цикл с отключённой сетью --------------------------------------------------------


@pytest.fixture
def blocked_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during offline cycle")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(scope="module")
def scenarios():
    return load_synthetic()


@pytest.fixture
def fire(scenarios):
    return scenarios["syn-001-fire-respond"]


def run_fire_cycle(fire, rubric):
    """Уведомление → карточка → решение → разговор → подтверждение → сдача → оценка."""
    s = TrainingSession(
        fire,
        channel=Channel.VOICE,
        tts=FakeTTS(),
        media=MockMediaTransport(),
        llm=FakeLLM(),
        judge=StubJudge(),
    )
    s.notify()
    s.clock.advance(20)
    visible = s.open_card()
    assert visible  # карточка действительно показана ученику
    s.edit_card(
        services=["fire_service", "ambulance"],
        description="Дым из окна квартиры соседей, пострадавших пока не выявлено.",
    )
    s.decide(DispatcherDecision.RESPOND)
    s.dial()
    s.say(FIRE_REPORT)
    turn = s.say("Да, верно.")
    assert turn.ack_id is not None  # подтверждение руководителя получено
    s.clock.advance(170)
    submitted = s.submit()
    assert submitted is not None
    return s, s.evaluate(rubric)


def test_full_offline_cycle_reaches_explained_score_and_recommendation(
    fire, rubric, blocked_network
):
    trainee_id = uuid4()
    attempts: list[AttemptRecord] = []
    last_evaluation = None
    for _ in range(3):  # G-3: уровень сегмента виден от 3 valid_score-попыток
        s, evaluation = run_fire_cycle(fire, rubric)
        last_evaluation = evaluation
        assert evaluation.summary.verdict is Verdict.PASSED  # оценка получена, не «сбой»
        # оценка с объяснением: у решённых критериев есть понятное обоснование для ученика
        explained = [r for r in evaluation.results if r.explanation]
        assert explained
        score_version = ScoreVersion(
            score_version_id=uuid4(),
            attempt_id=s.attempt_id,
            author="system",
            created_at=T0,
            criterion_results=evaluation.results,
            summary=evaluation.summary,
        )
        attempts.append(
            AttemptRecord(
                s.attempt_id, fire.dds_profile, "low", evaluation.track, score_version, rubric
            )
        )

    assert last_evaluation is not None and last_evaluation.summary.valid_score

    profile = build_profile(
        trainee_id,
        attempts,
        dds_profile=fire.dds_profile,
        difficulty_band="low",
        track=attempts[0].track,
        aggregation_rules=VersionRef(name="offline-cycle-rules", version="1"),
        now=T0,
    )
    assert profile.status is ProfileStatus.OK

    task_pool = TaskPool(
        pool_snapshot_id=uuid4(),
        dds_profile=fire.dds_profile,
        assigned_by="teacher-1",
        published_at=T0,
        tasks=(
            PoolTask(
                task_id="t0",
                dds_profile=fire.dds_profile,
                difficulty_band="low",
                difficulty_rank=0,
            ),
        ),
    )
    recommendation = recommend(
        trainee_id=trainee_id,
        profile=profile,
        pool=task_pool,
        current_rank=0,
        selection_rules=profile.aggregation_rules,
        now=T0,
    )
    assert recommendation.status is RecommendationStatus.AVAILABLE
    assert recommendation.task_id == "t0"
    assert recommendation.reason
