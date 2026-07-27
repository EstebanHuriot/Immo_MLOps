from fastapi import FastAPI, HTTPException

from src.predict import predict_one, features   

app = FastAPI(title="Real estate price prediction API")

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