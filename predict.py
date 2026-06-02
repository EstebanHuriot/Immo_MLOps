import joblib
import os
import pandas as pd

DATA_DIR = r'./data/annonces_france'
MODELS_DIR = r'./models/'

# trouver le modèle
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_INFO_PATH = os.path.join(MODELS_DIR, "feature_info.pkl")

# trouver les données
INPUT_FILE = os.path.join(DATA_DIR, 'nouvelles_annonces.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'predictions_prix_m2.csv')

# charger le modèle
model = joblib.load(MODEL_PATH)
feature_info = joblib.load(FEATURE_INFO_PATH)

features = feature_info["features"]


def predict_file(input_file, output_file):
    
    df = pd.read_csv(input_file, sep=";", low_memory=False)
    X = df[features]

    predictions = model.predict(X)

    df["prix_m2_vente_predit"] = predictions
    df.to_csv(output_file, sep=";", index=False, encoding="utf-8")

    return df


def predict_one(annonce: dict):

    """    colonnes_manquantes = []

    for col in features:
        if col not in annonce:
            colonnes_manquantes.append(col)

    if colonnes_manquantes:
        raise ValueError(
            "Colonnes manquantes : " + ", ".join(colonnes_manquantes)
        )
    """
    X = pd.DataFrame([annonce])
    X = X[features]

    prediction = model.predict(X)[0]

    return round(float(prediction), 2)



if __name__ == "__main__":
    INPUT_FILE = os.path.join(DATA_DIR, "nouvelles_annonces.csv")
    OUTPUT_FILE = os.path.join(DATA_DIR, "predictions_prix_m2.csv")

    result = predict_file(INPUT_FILE, OUTPUT_FILE)

    print(result[["prix_m2_vente_predit"]].head())