"""
Tests unitaires pour le microservice prediction-api.

Ils ne dependent pas d'un modele reellement entraine : predict_one et
get_features sont monkeypatchees directement dans le module
services.prediction_api.app (c'est la qu'elles sont importees et utilisees).
"""
import services.prediction_api.app as app_module
from services.prediction_api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_features_returns_503_when_no_model(monkeypatch):
    def _raise_not_found():
        raise FileNotFoundError("Modele introuvable.")

    monkeypatch.setattr(app_module, "get_features", _raise_not_found)
    response = client.get("/features")
    assert response.status_code == 503


def test_features_returns_list_when_model_present(monkeypatch):
    monkeypatch.setattr(app_module, "get_features", lambda: ["surface", "ville"])
    response = client.get("/features")
    assert response.status_code == 200
    assert response.json() == {"features": ["surface", "ville"]}


def test_predict_returns_503_when_no_model(monkeypatch):
    def _raise_not_found(annonce):
        raise FileNotFoundError("Modele introuvable.")

    monkeypatch.setattr(app_module, "predict_one", _raise_not_found)
    response = client.post("/predict", json={"surface": 50})
    assert response.status_code == 503


def test_predict_returns_400_on_bad_input(monkeypatch):
    def _raise_value_error(annonce):
        raise ValueError("Colonnes manquantes : surface")

    monkeypatch.setattr(app_module, "predict_one", _raise_value_error)
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_predict_returns_price_on_success(monkeypatch):
    monkeypatch.setattr(app_module, "predict_one", lambda annonce: 3500.0)
    response = client.post("/predict", json={"surface": 50})
    assert response.status_code == 200
    assert response.json() == {"prix": 3500.0}
