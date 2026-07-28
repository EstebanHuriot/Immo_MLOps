import joblib
import os
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "./data/annonces_france")
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_INFO_PATH = os.path.join(MODELS_DIR, "feature_info.pkl")

# --- Chargement paresseux (lazy load) ---
# Avant: le modele etait charge au niveau du module (au demarrage du process).
# Probleme: si models/best_model.pkl n'existe pas encore (repo fraichement clone,
# aucun entrainement lance), joblib.load() leve une FileNotFoundError au demarrage
# et le conteneur prediction-api crashe / boucle en restart avant meme de repondre
# a une requete. Avec le chargement paresseux, l'API demarre toujours, et renvoie
# une erreur claire (503) tant qu'aucun modele n'est disponible.
_model = None
_feature_info = None


def _load_artifacts():
    global _model, _feature_info
    if _model is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_INFO_PATH):
            raise FileNotFoundError(
                "Modele introuvable. Lance un entrainement "
                "(POST /train sur le training-api, ou le DAG Airflow) "
                "avant d'appeler /predict."
            )
        _model = joblib.load(MODEL_PATH)
        _feature_info = joblib.load(FEATURE_INFO_PATH)
    return _model, _feature_info


def get_features():
    _, feature_info = _load_artifacts()
    return feature_info["features"]


def predict_file(input_file, output_file):
    model, feature_info = _load_artifacts()
    features = feature_info["features"]
    var_cat = feature_info.get("var_cat", [])
    df = pd.read_csv(input_file, sep=";", low_memory=False)
    X = df[features]
    if var_cat:
        X[var_cat] = X[var_cat].astype(str)
    predictions = model.predict(X)
    df["prix_m2_vente_predit"] = predictions
    df.to_csv(output_file, sep=";", index=False, encoding="utf-8")
    return df


def predict_one(annonce: dict):
    model, feature_info = _load_artifacts()
    features = feature_info["features"]
    var_cat = feature_info.get("var_cat", [])
    colonnes_manquantes = [col for col in features if col not in annonce]
    if colonnes_manquantes:
        raise ValueError("Colonnes manquantes : " + ", ".join(colonnes_manquantes))
    X = pd.DataFrame([annonce])[features]
    # Meme conversion de type que celle appliquee a l'entrainement
    # (cf. train.py:prepare_model_data) pour rester coherent avec les
    # categories_ apprises par OneHotEncoder.
    if var_cat:
        X[var_cat] = X[var_cat].astype(str)
    prediction = model.predict(X)[0]
    return round(float(prediction), 2)


if __name__ == "__main__":
    input_file = os.path.join(DATA_DIR, "nouvelles_annonces.csv")
    output_file = os.path.join(DATA_DIR, "predictions_prix_m2.csv")
    result = predict_file(input_file, output_file)
    print(result[["prix_m2_vente_predit"]].head())
