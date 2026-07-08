import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


import warnings
import joblib

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")


# =========================
# CONFIGURATION
# =========================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "annonces_france"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "prix_m2_vente"

N_SAMPLE = 100_000

BEST_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("Model_immo")




# =========================
# FONCTIONS
# =========================

def load_main_dataset():
    df = pd.read_csv(
        DATA_DIR / "df_france_ventes_cleaned.csv",
        sep=";",
        low_memory=False,
    )

    if N_SAMPLE is not None and len(df) > N_SAMPLE:
        df = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE)

    print(f"Dataset chargé : {len(df):,} lignes, {len(df.columns)} colonnes")
    return df


def prepare_model_data(df):
    cols_drop = [
        "Unnamed: 0",
        "index",
        "idannonce",
        "date",
        "prix_bien",
        "mensualiteFinance",
        "DEP_SOURCE",
    ]

    cols_drop = [col for col in cols_drop if col in df.columns]

    df_model = df.drop(columns=cols_drop)

    if TARGET not in df_model.columns:
        raise ValueError(f"La variable cible '{TARGET}' est absente du dataset.")

    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]

    var_num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    var_cat = [
        col for col in X.select_dtypes(include="object").columns
        if X[col].nunique() <= 110
    ]

    features = var_num + var_cat
    X = X[features]

    print(f"Features numériques : {len(var_num)}")
    print(f"Features catégorielles : {len(var_cat)}")
    print(f"Total features : {len(features)}")

    return X, y, var_num, var_cat, features


def build_pipeline(var_num, var_cat):
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), var_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), var_cat),
    ])

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(**BEST_PARAMS)),
    ])

    return pipeline


def train_and_evaluate(X, y, var_num, var_cat):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print(f"Train : {len(X_train):,} lignes")
    print(f"Test  : {len(X_test):,} lignes")

    pipeline = build_pipeline(var_num, var_cat)

    print("Entraînement du modèle...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Performance du modèle :")
    print(f"RMSE : {rmse:.2f} €/m²")
    print(f"MAE  : {mae:.2f} €/m²")
    print(f"R²   : {r2:.4f}")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return pipeline, metrics


def save_artifacts(pipeline, feature_info):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.pkl"
    feature_info_path = MODELS_DIR / "feature_info.pkl"

    joblib.dump(pipeline, model_path)
    joblib.dump(feature_info, feature_info_path)

    print(f"Modèle sauvegardé : {model_path}")
    print(f"Infos features sauvegardées : {feature_info_path}")


# =========================
# MAIN
# =========================

def train_model():
    df = load_main_dataset()

    X, y, var_num, var_cat, features = prepare_model_data(df)

    feature_info = {
        "var_num": var_num,
        "var_cat": var_cat,
        "features": features,
        "target": TARGET,
    }

    with mlflow.start_run(run_name="random_forest_immo_train"):

        # Logger les paramètres principaux
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("n_sample", N_SAMPLE)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_features_num", len(var_num))
        mlflow.log_param("n_features_cat", len(var_cat))

        pipeline, metrics = train_and_evaluate(
            X=X,
            y=y,
            var_num=var_num,
            var_cat=var_cat,
        )

        feature_info.update(metrics)

        # Sauvegarde locale avec joblib
        save_artifacts(pipeline, feature_info)

        # Logger les métriques dans MLflow
        mlflow.log_metrics({
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "r2": float(metrics["r2"]),
            "n_train": int(metrics["n_train"]),
            "n_test": int(metrics["n_test"]),
        })

        # Logger les fichiers locaux comme artifacts
        mlflow.log_artifact(str(MODELS_DIR / "best_model.pkl"))
        mlflow.log_artifact(str(MODELS_DIR / "feature_info.pkl"))

        # Logger le modèle complet sklearn dans MLflow
        mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        input_example=X.head(1),
        serialization_format="cloudpickle",
            )

        print("Modèle loggé dans MLflow")

        return {
            "message": "Entraînement terminé",
            "metrics": {
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
                "n_train": int(metrics["n_train"]),
                "n_test": int(metrics["n_test"]),
            }
        }

    save_artifacts(pipeline, feature_info)

    return {
        "message": "Entraînement terminé",
        "metrics": {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "r2": float(metrics["r2"]),
            "n_train": int(metrics["n_train"]),
            "n_test": int(metrics["n_test"]),
        }
    }


if __name__ == "__main__":
    result = train_model()
    print(result)