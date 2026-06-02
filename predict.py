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

print("Modèle chargé")
print(f"Nombre de features attendues : {len(features)}")

# charger les données
df = pd.read_csv(INPUT_FILE, sep=";", low_memory=False)

print(f"Fichier chargé : {len(df)} lignes")

# vérifier s'il y a des colonnes manquantes
colonnes_manquantes = []

for col in features:
    if col not in df.columns:
        colonnes_manquantes.append(col)

if colonnes_manquantes:
    raise ValueError(
        "Certaines colonnes nécessaires au modèle sont absentes : "
        + ", ".join(colonnes_manquantes)
    )


# garder uniquement les colonnes utilisées par le modèle
X = df[features]


# prédire
predictions = model.predict(X)


# sauvegarder le résultat
df["prix_m2_vente_predit"] = predictions

df.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8")

print("Prédictions terminées")
print(f"Résultat sauvegardé ici : {OUTPUT_FILE}")

print("\nAperçu :")
print(df[["prix_m2_vente_predit"]].head())