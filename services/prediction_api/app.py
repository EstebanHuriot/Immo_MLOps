from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge

from src.predict import predict_one, get_features

# kpis for prometheus (average predicted price and number of predictions)
total_predicted_price = 0.0
total_predictions = 0
prediction_count = Counter("model_predictions_total","Nombre total de prédictions effectuées")
average_predicted_price = Gauge("model_average_predicted_price_euros","Prix moyen prédit par le modèle en euros")

# drift monitoring on surface
surface_drift = Gauge("model_surface_drift_ratio","Ecart relatif entre la surface moyenne en production et la surface moyenne d'entrainement")
TRAIN_SURFACE_MEAN = 75.0
total_surface = 0.0
surface_count = 0



app = FastAPI(title="Real estate price prediction API")

Instrumentator().instrument(app).expose(app) # prometheus 


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
    global total_predicted_price, total_predictions
    global total_surface, surface_count

    try:
        # Vérifie d'abord la surface
        surface = float(annonce["surface"])

        # Prédiction
        prediction = predict_one(annonce)

        # KPI predictions
        prediction_count.inc()
        total_predictions += 1
        total_predicted_price += float(prediction)

        average_predicted_price.set(
            total_predicted_price / total_predictions
        )

        # KPI drift sur la surface
        surface_count += 1
        total_surface += surface

        prod_surface_mean = total_surface / surface_count

        drift_ratio = (
            abs(prod_surface_mean - TRAIN_SURFACE_MEAN)
            / TRAIN_SURFACE_MEAN
        )

        surface_drift.set(drift_ratio)

        return {"prix": prediction}

    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="La variable 'surface' est manquante"
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
