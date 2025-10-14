# tests/test_api.py
from fastapi.testclient import TestClient
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app


client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

def test_predict():
    payload = {
      "wrestler_1_win_rate": 0.8,
      "wrestler_2_win_rate": 0.55,
      "storyline_rivalry": 1,
      "recent_form": 3
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "prediction" in j
    assert "probability" in j
