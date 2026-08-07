from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.predict import predict_one, get_features

app = FastAPI(title="Real estate price prediction API")


@app.get("/")
def home():
    return {"message": "API de prediction du prix de l'immobilier"}

# link between front and back
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/features")
def features_endpoint():
    try:
        return {"features": get_features()}
    except FileNotFoundError as e:
        # Pas encore de modele entraine -> 503 (service pas encore pret),
        # au lieu de faire planter le conteneur au demarrage.
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict")
def predict(annonce: dict):
    try:
        prediction = predict_one(annonce)
        return {"prix": prediction}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
