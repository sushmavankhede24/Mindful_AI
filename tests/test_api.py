import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.parametrize(
    "prompt,expected_categories",
    [
        ("I am having trouble sleeping", ["sleep", "relax"]),
        ("I need to concentrate for my study", ["focus"]),
        ("I am feeling stressed and anxious", ["stress"]),
        ("I want to relax after work", ["relax", "focus"]),
        ("I have a job interview tomorrow", ["confidence"]),
    ],
)
def test_recommend_categories(prompt, expected_categories):
    response = client.post("/recommend", json={"prompt": prompt})
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "recommendation" in data
    assert data["category"].lower() in expected_categories


def test_recommend_unknown():
    payload = {"prompt": "This is totally unrelated"}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"].lower() == "unknown"
    assert "no match found" in data["recommendation"].lower()


def test_recommend_empty():
    payload = {"prompt": ""}
    response = client.post("/recommend", json={"prompt": payload})
    assert response.status_code == 422
