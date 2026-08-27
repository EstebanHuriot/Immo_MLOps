import os
import secrets
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator

from src.train import train_model

app = FastAPI(title="Real estate training API")

Instrumentator().instrument(app).expose(app)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def verify_api_key(api_key: str = Depends(api_key_header)):
    expected_api_key = os.getenv("API_KEY")

    if not expected_api_key:
        raise HTTPException(status_code=500, detail="API key is not configured")

    if not api_key or not secrets.compare_digest(api_key, expected_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def home():
    return {"message":"API de prédiction du prix de l'immobilier"}


@app.post("/train")
def train(_: str = Depends(verify_api_key)):
    try:
        result = train_model()

        return { **result, "note": "Le modèle a été sauvegardé. Redémarre le serveur API pour utiliser le nouveau modèle."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )