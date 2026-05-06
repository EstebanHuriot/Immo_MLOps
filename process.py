import os
import gc
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


# =========================
# CONFIGURATION
# =========================

DATA_DIR = "./data/annonces_france"

INPUT_FILE = f"{DATA_DIR}/df_france_ventes.csv"
OUTPUT_FILE = f"{DATA_DIR}/df_france_ventes_cleaned.csv"

TARGET = "prix_m2_vente"
SEUIL_NA = 0.40


# =========================
# FONCTIONS
# =========================

def remove_outliers_iqr(df, features, factor=1.5):
    """
    Cap les outliers par méthode IQR.
    On ne supprime pas les lignes, on borne les valeurs extrêmes.
    """
    df_capped = df.copy()

    for feature in features:
        if feature not in df_capped.columns:
            continue

        Q1 = df_capped[feature].quantile(0.25)
        Q3 = df_capped[feature].quantile(0.75)
        IQR = Q3 - Q1

        if pd.isna(IQR) or IQR == 0:
            continue

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        n_outliers = ((df_capped[feature] < lower) | (df_capped[feature] > upper)).sum()

        df_capped[feature] = df_capped[feature].clip(lower, upper)

        if n_outliers > 0:
            print(f"  {feature}: {n_outliers:,} outliers traités")

    return df_capped


def main():
    print("=" * 60)
    print("PREPROCESSING DONNÉES FRANCE")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_FILE}")

    # =========================
    # 1. CHARGEMENT
    # =========================

    df = pd.read_csv(INPUT_FILE, sep=";", low_memory=False)
    print(f"Données chargées: {len(df):,} lignes, {len(df.columns)} colonnes")

    # =========================
    # 2. CONVERSION TYPES NUMÉRIQUES
    # =========================

    numeric_cols_to_check = [
        "surface",
        "prix_bien",
        "prix_m2_vente",
        "duree_int",
        "dpeC",
    ]

    for col in numeric_cols_to_check:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # 3. SUPPRESSION COLONNES TROP MANQUANTES
    # =========================

    cols_na_drop = df.columns[df.isna().mean() > SEUIL_NA].tolist()

    # On évite de supprimer la cible si jamais elle dépasse le seuil
    if TARGET in cols_na_drop:
        cols_na_drop.remove(TARGET)

    print(f"Colonnes supprimées pour NA > {SEUIL_NA * 100:.0f}%: {len(cols_na_drop)}")
    if cols_na_drop:
        print(cols_na_drop)
        df = df.drop(columns=cols_na_drop)

    # =========================
    # 4. SUPPRESSION COLONNES INUTILES / LEAKAGE
    # =========================

    cols_exclude = [
        "idannonce",
        "Unnamed: 0",
        "index",
        "prix_bien",
        "mensualiteFinance",
        "DEP_SOURCE",
    ]

    cols_exclude = [c for c in cols_exclude if c in df.columns]

    print(f"Colonnes inutiles / leakage supprimées: {cols_exclude}")

    if cols_exclude:
        df = df.drop(columns=cols_exclude)

    # =========================
    # 5. NETTOYAGE VARIABLE CIBLE
    # =========================

    if TARGET not in df.columns:
        raise ValueError(f"Variable cible absente: {TARGET}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df[TARGET] = df[TARGET].replace([np.inf, -np.inf], np.nan)

    if "surface" in df.columns:
        df["surface"] = pd.to_numeric(df["surface"], errors="coerce")
        df = df[df["surface"] > 0]

    before = len(df)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    after = len(df)

    print(f"Lignes supprimées cible invalide / surface invalide: {before - after:,}")
    print(f"Lignes restantes: {len(df):,}")

    # =========================
    # 6. IMPUTATION SPÉCIFIQUE
    # =========================

    if "duree_int" in df.columns:
        median_duree = df["duree_int"].median()
        df["duree_int"] = df["duree_int"].fillna(median_duree)
        print(f"duree_int imputée par médiane: {median_duree}")

    if "logement_neuf" in df.columns:
        df["logement_neuf"] = df["logement_neuf"].fillna("n")
        print("logement_neuf imputée par 'n'")

    if "dpeC" in df.columns:
        median_dpe = df["dpeC"].median()
        df["dpeC"] = df["dpeC"].fillna(median_dpe)
        print(f"dpeC imputée par médiane: {median_dpe}")

    if "ges_class" in df.columns:
        mode_ges = df["ges_class"].mode(dropna=True)
        df["ges_class"] = df["ges_class"].fillna(
            mode_ges[0] if len(mode_ges) > 0 else "Unknown"
        )
        print("ges_class imputée par mode")

    # =========================
    # 7. IMPUTATION GÉNÉRALE RESTANTE
    # =========================

    missing_before = df.isna().sum().sum()
    print(f"Valeurs manquantes restantes avant imputation générale: {missing_before:,}")

    for col in df.columns:
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(
                    mode_val[0] if len(mode_val) > 0 else "Unknown"
                )

    missing_after = df.isna().sum().sum()
    print(f"Valeurs manquantes après imputation: {missing_after:,}")

    # =========================
    # 8. TRAITEMENT OUTLIERS
    # =========================

    var_num = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    exclude_from_outliers = [
        "mapCoordonneesLatitude",
        "mapCoordonneesLongitude",
        "INSEE_COM",
        "IRIS",
        "CODE_IRIS",
        "REG",
        "DEP",
        "GRD_QUART",
        "UU2010",
    ]

    var_to_cap = [
        v for v in var_num
        if v not in exclude_from_outliers
    ]

    print(f"Traitement outliers sur {len(var_to_cap)} variables:")
    df = remove_outliers_iqr(df, var_to_cap)

    # =========================
    # 9. VARIABLES NUM / CAT
    # =========================

    var_num = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    var_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if TARGET in var_num:
        var_num.remove(TARGET)

    print(f"Variables numériques: {len(var_num)}")
    print(f"Variables catégorielles: {len(var_cat)}")

    # =========================
    # 10. SAUVEGARDE
    # =========================

    df.to_csv(OUTPUT_FILE, index=False, sep=";")

    print("\n" + "=" * 60)
    print("PREPROCESSING TERMINÉ")
    print("=" * 60)
    print(f"Fichier sauvegardé: {OUTPUT_FILE}")
    print(f"Lignes finales: {len(df):,}")
    print(f"Colonnes finales: {len(df.columns)}")
    print(f"Variable cible: {TARGET}")

    gc.collect()


if __name__ == "__main__":
    main()