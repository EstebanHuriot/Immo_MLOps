import gc
import pandas as pd
import requests
import os
from io import StringIO
from tqdm import tqdm
import time

BASE_URL = "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/annonces_git"

#BASE_URL = r"https://github.com/klopstock-dviz/immo_vis/tree/master/data/annonces_git"


# liste des 101 départements
DEPARTEMENTS_METROPOLE = [f"{i:02d}" for i in range(1, 96) if i != 20]
DEPARTEMENTS_CORSE = ["2A", "2B"]
DEPARTEMENTS_DOMTOM = ["971", "972", "973", "974", "976"]

TOUS_DEPARTEMENTS = DEPARTEMENTS_METROPOLE + DEPARTEMENTS_CORSE + DEPARTEMENTS_DOMTOM


# téléchargement d'un CSV depuis github
def fetch_csv_from_github(filename, base_url=BASE_URL):
    """
    Télécharge un fichier CSV depuis GitHub et retourne un DataFrame.
    """
    url = f"{base_url}/{filename}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        content = response.text
        try:
            df = pd.read_csv(StringIO(content), sep=';', low_memory=False)
            if len(df.columns) <= 1:
                df = pd.read_csv(StringIO(content), sep=',', low_memory=False)
        except Exception:
            df = pd.read_csv(StringIO(content), sep=',', low_memory=False)
        
        return df
    except requests.exceptions.RequestException as e:
        print(f"Erreur pour {filename}: {e}")
        return None    

def main():

    OUTPUT_DIR = "../data/annonces_france"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dossier de sortie: {OUTPUT_DIR}")
    # téléchargement par batch de 10 départements
    BATCH_SIZE = 10
    OUTPUT_FILE = f"{OUTPUT_DIR}/df_france_ventes.csv"

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Fichier existant supprimé: {OUTPUT_FILE}")

    departements_ok = []
    departements_ko = []
    total_lignes = 0
    header_written = False

    print(f"Téléchargement des fichiers VENTES par batch de {BATCH_SIZE}...\n")

    for batch_start in range(0, len(TOUS_DEPARTEMENTS), BATCH_SIZE):
        batch_deps = TOUS_DEPARTEMENTS[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(TOUS_DEPARTEMENTS) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n--- Batch {batch_num}/{total_batches} ({batch_deps}) ---")

        dfs_batch = []

        for dep in tqdm(batch_deps, desc=f"Batch {batch_num}"):
            filename = f"df_annonces_gps_iris_ventes_{dep}.csv"
            df = fetch_csv_from_github(filename)

            if df is not None and len(df) > 0:
                df['DEP_SOURCE'] = dep
                dfs_batch.append(df)
                departements_ok.append(dep)
            else:
                departements_ko.append(dep)

            time.sleep(0.1)

        if dfs_batch:
            df_batch = pd.concat(dfs_batch, ignore_index=True)
            batch_lignes = len(df_batch)
            total_lignes += batch_lignes

            df_batch.to_csv(OUTPUT_FILE, index=False, sep=';', 
                            mode='a', header=not header_written)
            header_written = True

            print(f"  -> {batch_lignes:,} lignes sauvegardées (total: {total_lignes:,})")

            del dfs_batch, df_batch
            gc.collect()

    print(f"\n\n{'='*60}")
    print(f"TÉLÉCHARGEMENT TERMINÉ")
    print(f"{'='*60}")
    print(f"  - Départements OK: {len(departements_ok)}")
    print(f"  - Départements KO: {len(departements_ko)}")
    print(f"  - Total lignes: {total_lignes:,}")
    if departements_ko:
        print(f"  - Liste KO: {departements_ko}")


    # vérification du fichier final
    if os.path.exists(OUTPUT_FILE):
        print(f"Fichier créé: {OUTPUT_FILE}")
        print(f"Taille: {os.path.getsize(OUTPUT_FILE) / 1024**2:.1f} MB")

        chunk_iter = pd.read_csv(OUTPUT_FILE, sep=';', chunksize=10000)
        first_chunk = next(chunk_iter)
        colonnes = first_chunk.columns.tolist()
        n_lignes = len(first_chunk)

        for chunk in chunk_iter:
            n_lignes += len(chunk)

        print(f"Lignes totales: {n_lignes:,}")
        print(f"Colonnes ({len(colonnes)}): {colonnes}")
    else:
        print("Aucun fichier créé")

    print("collecte terminée")

if __name__ == "__main__":
    main()