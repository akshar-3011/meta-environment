"""API tests for pipeline FastAPI endpoints."""

from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

pipeline_api = importlib.import_module("workplace_env.api.pipeline_app")


def test_health_endpoint_returns_ok():
    client = TestClient(pipeline_api.app, raise_server_exceptions=False)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "ok"


def test_infer_endpoint_returns_actions_for_valid_request():
    client = TestClient(pipeline_api.app, raise_server_exceptions=False)

    response = client.post(
        "/infer",
        json={
            "email": "My issue is unresolved.",
            "strategy": "enhanced",
            "scenario_difficulty": "medium",
            "urgency": "high",
            "sentiment": "negative",
            "complexity_score": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["score"] == 1.0
    assert payload["breakdown"]["action_count"] == 3


def test_infer_endpoint_validation_error_for_missing_email():
    client = TestClient(pipeline_api.app)

    response = client.post(
        "/infer",
        json={
            "strategy": "standard",
            "scenario_difficulty": "easy",
            "urgency": "medium",
            "sentiment": "neutral",
            "complexity_score": 2,
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_grade_endpoint_success_for_reply_action():
    client = TestClient(pipeline_api.app)

    response = client.post(
        "/grade",
        json={
            "action_type": "reply",
            "content": "We are sorry and will resolve this quickly.",
            "actual_category": "complaint",
            "step_count": 2,
            "scenario_difficulty": "medium",
            "min_reply_length": 30,
            "previous_actions": {"classify": 0.9},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["breakdown"]["action_type"] == "reply"


def test_pipeline_endpoint_success_returns_three_steps():
    client = TestClient(pipeline_api.app)

    response = client.post(
        "/pipeline",
        json={
            "email": "Please help, this has been frustrating.",
            "actual_category": "complaint",
            "strategy": "standard",
            "scenario_difficulty": "easy",
            "min_reply_length": 30,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["breakdown"]["total_steps"] == 3
    assert len(payload["breakdown"]["steps"]) == 3


def test_pipeline_endpoint_surfaces_inference_error(monkeypatch):
    class _BrokenStrategy:
        def build_actions(self, _observation):
            raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_api, "_select_strategy", lambda _strategy: _BrokenStrategy())

    client = TestClient(pipeline_api.app, raise_server_exceptions=False)
    response = client.post(
        "/pipeline",
        json={
            "email": "Need urgent help",
            "actual_category": "complaint",
            "strategy": "standard",
            "scenario_difficulty": "easy",
            "min_reply_length": 30,
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "INFERENCE_ERROR"
