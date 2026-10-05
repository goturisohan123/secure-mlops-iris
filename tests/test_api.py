import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app, infer

TEST_KEY = "unit-test-key-only-not-a-deployment-secret"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_KEY)

    with TestClient(app) as test_client:
        yield test_client


def headers():
    return {"X-API-Key": TEST_KEY}


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_missing_key(client):
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )

    assert response.status_code == 401


def test_wrong_key(client):
    response = client.post(
        "/predict",
        headers={"X-API-Key": "incorrect"},
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )

    assert response.status_code == 401


def test_prediction(client):
    response = client.post(
        "/predict",
        headers=headers(),
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )

    assert response.status_code == 200
    assert response.json()["species"] == "setosa"
    assert response.json()["model_version"]


@pytest.mark.parametrize(
    "features",
    [
        [5.1, 3.5],
        [-1, 3.5, 1.4, 0.2],
        [100, 3.5, 1.4, 0.2],
        ["invalid", 3.5, 1.4, 0.2],
    ],
)
def test_invalid_input(client, features):
    response = client.post(
        "/predict",
        headers=headers(),
        json={"features": features},
    )

    assert response.status_code == 422


def test_unexpected_field(client):
    response = client.post(
        "/predict",
        headers=headers(),
        json={
            "features": [5.1, 3.5, 1.4, 0.2],
            "unexpected": "rejected",
        },
    )

    assert response.status_code == 422


def test_stats_requires_auth(client):
    response = client.get("/stats")

    assert response.status_code == 401


def test_export_matches_sklearn():
    cases = json.loads(
        Path("models/parity_cases.json").read_text()
    )

    for features, expected_class, expected_probabilities in zip(
        cases["features"],
        cases["predictions"],
        cases["probabilities"],
    ):
        predicted, probabilities = infer(features)

        assert predicted == expected_class
        assert probabilities == pytest.approx(
            expected_probabilities
        )
