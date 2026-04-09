"""Tests for the health check endpoint — ensures Docker HEALTHCHECK passes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from workplace_env.api.app import app


client = TestClient(app)


def test_healthcheck_endpoint_responds():
    """Docker HEALTHCHECK endpoint returns 200 with status=healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"


def test_root_landing_page_responds():
    """GET / returns a 200 with HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Workplace Env" in response.text


def test_reset_endpoint_responds():
    """POST /reset returns 200 with observation data."""
    response = client.post("/reset", json={})
    assert response.status_code == 200
    data = response.json()
    assert "observation" in data or "email" in data.get("observation", {})


def test_step_after_reset_responds():
    """POST /step after /reset returns reward and observation."""
    client.post("/reset", json={})
    response = client.post("/step", json={
        "action": {"action_type": "classify", "content": "refund"}
    })
    assert response.status_code == 200
    data = response.json()
    assert "reward" in data or data.get("reward") is not None


def test_state_endpoint_responds():
    """GET /state returns current environment state."""
    client.post("/reset", json={})
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert "step_count" in data


def test_full_episode_lifecycle():
    """A complete 3-step episode produces done=true at step 3.

    Uses the environment directly because OpenEnv's create_app
    may not share state across stateless HTTP requests.
    """
    from workplace_env.environment.workplace_environment import WorkplaceEnvironment
    from workplace_env.models import WorkplaceAction

    env = WorkplaceEnvironment()
    env.reset()

    obs1 = env.step(WorkplaceAction(action_type="classify", content="refund"))
    assert obs1.done is False

    obs2 = env.step(WorkplaceAction(action_type="reply", content="Thank you."))
    assert obs2.done is False

    obs3 = env.step(WorkplaceAction(action_type="escalate", content="no"))
    assert obs3.done is True
