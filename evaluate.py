import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DATA_DIR = "./data/annonces_france"
MODELS_DIR = "./models"

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_INFO_PATH = os.path.join(MODELS_DIR, "feature_info.pkl")

INPUT_FILE = os.path.join(DATA_DIR, "nouvelles_annonces.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "evaluation_predictions.csv")


# Charger le modèle et les infos
model = joblib.load(MODEL_PATH)
feature_info = joblib.load(FEATURE_INFO_PATH)

features = feature_info["features"]
target = feature_info["target"]

print("Modèle chargé")


# Charger les données à évaluer
df = pd.read_csv(INPUT_FILE, sep=";", low_memory=False)

print(f"Fichier chargé : {len(df)} lignes")


# Vérifier que la cible existe
if target not in df.columns:
    raise ValueError(f"Colonne cible absente : {target}")


# Préparer X et y
X = df[features]
y_true = df[target]


# Prédire
y_pred = model.predict(X)


# Calculer les métriques
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)


# Afficher les résultats
print("\nRésultats :")
print(f"RMSE : {rmse:.2f} €/m²")
print(f"MAE  : {mae:.2f} €/m²")
print(f"R²   : {r2:.4f}")


# Sauvegarder un fichier avec les prédictions
df["prix_m2_vente_predit"] = y_pred
df["erreur"] = df["prix_m2_vente_predit"] - df[target]
df["erreur_absolue"] = abs(df["erreur"])

df.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8")

print(f"\nFichier sauvegardé : {OUTPUT_FILE}")