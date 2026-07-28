"""
Tests unitaires pour le microservice training-api.

train_model est monkeypatche directement dans le module
services.training_api.app pour ne jamais lancer un vrai entrainement
(couteux, depend de data/annonces_france/*.csv) pendant la CI.
"""
import services.training_api.app as app_module
from services.training_api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_train_success(monkeypatch):
    fake_result = {
        "message": "Entrainement termine",
        "metrics": {"rmse": 100.0, "mae": 80.0, "r2": 0.85, "n_train": 800, "n_test": 200},
    }
    monkeypatch.setattr(app_module, "train_model", lambda: fake_result)

    response = client.post("/train")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Entrainement termine"
    assert body["metrics"]["rmse"] == 100.0
    assert "note" in body


def test_train_failure_returns_500(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("dataset manquant")

    monkeypatch.setattr(app_module, "train_model", _raise)

    response = client.post("/train")
    assert response.status_code == 500
