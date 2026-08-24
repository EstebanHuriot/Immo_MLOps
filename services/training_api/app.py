from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.train import train_model

app = FastAPI(title="Real estate training API")

Instrumentator().instrument(app).expose(app) # prometheus

@app.get("/")
def home():
    return {"message":"API de prédiction du prix de l'immobilier"}


@app.post("/train")
def train():
    try:
        result = train_model()
    
        return { **result, "note": "Le modèle a été sauvegardé. Redémarre le serveur API pour utiliser le nouveau modèle."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )