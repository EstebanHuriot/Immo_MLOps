import os
import pandas as pd

DATA_DIR = "./data/annonces_france"

INPUT_FILE = os.path.join(DATA_DIR, "df_france_ventes_cleaned.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "nouvelles_annonces.csv")

N_ROWS = 1000
RANDOM_STATE = 42

df = pd.read_csv(INPUT_FILE, sep=";", low_memory=False)

df_subset = df.sample(n=N_ROWS, random_state=RANDOM_STATE)

df_subset.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8")

print(f"Fichier créé : {OUTPUT_FILE}")
print(f"Nombre de lignes : {len(df_subset)}")