from fastapi import FastAPI, HTTPException

from src.predict import predict_one, features
from src.train import train_model

app = FastAPI(title="Price Prediction API")

@app.get("/")
def home():
    return {"message":"API de prédiction du prix de l'immobilier"}


@app.get("/features")
def get_features():
    return {
        "features": features
    }


@app.post("/predict")
def predict(annonce:dict):
    try:
        prediction = predict_one(annonce)
        
        return {"prix":prediction}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
