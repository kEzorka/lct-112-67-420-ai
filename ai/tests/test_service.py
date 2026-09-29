"""HTTP-сервис: контракты и полный цикл попытки через API (без моделей — fallback)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from dds_ai.service import create_app

REPORT = (
    "Пожар в жилом доме, ул. Тестовая, д. 12, кв. 5. Дым из окна квартиры. "
    "ДДС реагирует, направляем пожарную охрану. Прошу принять доклад."
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("DDS_LLM_MODEL_PATH", raising=False)
    return TestClient(create_app())


def test_health_and_contracts(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm"] is None
    names = client.get("/v1/contracts").json()["schemas"]
    assert "attempt-event" in names
    assert client.get("/v1/contracts/criterion-result").json()["title"] == "CriterionResult"
    assert client.get("/v1/contracts/nope").status_code == 404


def test_attempt_cycle_over_http(client):
    scenarios = client.get("/v1/scenarios").json()
    assert all(s["synthetic"] for s in scenarios)
    r = client.post("/v1/attempts", json={"scenario_id": "syn-001-fire-respond", "card_seed": 1})
    assert r.status_code == 201
    aid = r.json()["attempt_id"]
    assert client.post(f"/v1/attempts/{aid}/open").json()["card"]
    client.post(f"/v1/attempts/{aid}/card", json={"values": {"services": ["fire_service"]}})
    client.post(f"/v1/attempts/{aid}/decision", json={"decision": "respond"})
    assert client.post(f"/v1/attempts/{aid}/dial").json()["connected"]
    client.post(f"/v1/attempts/{aid}/say", json={"text": REPORT})
    turn = client.post(f"/v1/attempts/{aid}/say", json={"text": "Да, верно."}).json()
    assert turn["reply"]["mode"] == "fallback"  # без LLM — проверенный шаблон
    assert client.post(f"/v1/attempts/{aid}/submit", json={}).status_code == 200
    ev = client.get(f"/v1/attempts/{aid}/evaluation").json()
    assert ev["summary"]["verdict"] in {"passed", "not_passed", "provisional"}
    assert ev["version_snapshot"]["attempt_id"] == aid
    types = [e["type"] for e in client.get(f"/v1/attempts/{aid}/events").json()]
    assert types[0] == "notification_shown" and "submit_accepted" in types


def test_unknown_attempt_and_scenario(client):
    assert client.post("/v1/attempts", json={"scenario_id": "nope"}).status_code == 404
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/v1/attempts/{missing}/events").status_code == 404
