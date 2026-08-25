import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
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
N_SAMPLE = int(os.environ.get("N_SAMPLE", 1000))

def _env_int_or_none(name, default):
    value = os.environ.get(name, default)
    if value in (None, "", "None"):
        return None
    return int(value)


# n_estimators / max_depth configurables pour pouvoir lancer facilement un run
# "baseline" (rapide, volontairement plus simple) avant le run final, et
# comparer les deux dans MLflow/DagsHub (onglet Experiments -> Compare).
BEST_PARAMS = dict(
    n_estimators=int(os.environ.get("N_ESTIMATORS", 20)),
    max_depth=_env_int_or_none("MAX_DEPTH", None),
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=2,
)

MLFLOW_RUN_NAME = os.environ.get("MLFLOW_RUN_NAME", "random_forest_immo_train")

# --- Tracking MLflow, configurable par variables d'environnement ---
# Par defaut : store local sur fichiers (mlruns/), pratique en dev.
#
# Pour centraliser sur DagsHub, deux options :
#
# 1) Client DagsHub (recommande, auth via navigateur, pas de token a copier) :
#      export DAGSHUB_REPO_OWNER=<ton_user_dagshub>
#      export DAGSHUB_REPO_NAME=<nom_du_repo_dagshub>
#
# 2) Ou manuellement, sans le client dagshub :
#      export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
#      export MLFLOW_TRACKING_USERNAME=<user ou token>
#      export MLFLOW_TRACKING_PASSWORD=<token DagsHub>
DAGSHUB_REPO_OWNER = os.environ.get("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.environ.get("DAGSHUB_REPO_NAME")

if DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME:
    import dagshub
    # Ouvre une page d'authentification dans le navigateur au premier lancement,
    # puis met en cache les identifiants localement pour les prochaines fois.
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file:{MLRUNS_DIR}")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Model_immo")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


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
    print(f"Dataset charge : {len(df):,} lignes, {len(df.columns)} colonnes")
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

    # Certaines colonnes categorielles melangent texte / nombres / NaN
    # (ex: codes IRIS). Ca fait planter OneHotEncoder a l'inference
    # (TypeError: ufunc 'isnan' not supported...) car les categories_
    # apprises ont un dtype mixte. On force un type texte homogene ici,
    # a l'entrainement, pour que ce soit coherent avec predict_one qui
    # fait la meme conversion cote API.
    X[var_cat] = X[var_cat].astype(str)

    print(f"Features numeriques : {len(var_num)}")
    print(f"Features categorielles : {len(var_cat)}")
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
    print("Entrainement du modele...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Performance du modele :")
    print(f"RMSE : {rmse:.2f} EUR/m2")
    print(f"MAE  : {mae:.2f} EUR/m2")
    print(f"R2   : {r2:.4f}")

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
    print(f"Modele sauvegarde : {model_path}")
    print(f"Infos features sauvegardees : {feature_info_path}")


# =========================
# MAIN
# =========================
def train_model():
    df = load_main_dataset()
    X, y, var_num, var_cat, features = prepare_model_data(df)

    # default values for the frontend bit.
    default_values = {}

    for col in var_num:
        default_values[col] = float(X[col].mean())

    for col in var_cat:
        default_values[col] = X[col].mode()[0]

    feature_info = {
        "var_num": var_num,
        "var_cat": var_cat,
        "features": features,
        "target": TARGET,
        "default_values": default_values,
    }

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
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

        save_artifacts(pipeline, feature_info)

        mlflow.log_metrics({
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "r2": float(metrics["r2"]),
            "n_train": int(metrics["n_train"]),
            "n_test": int(metrics["n_test"]),
        })

        mlflow.log_artifact(str(MODELS_DIR / "best_model.pkl"))
        mlflow.log_artifact(str(MODELS_DIR / "feature_info.pkl"))

    #    mlflow.sklearn.log_model(
    #        sk_model=pipeline,
    #        name="model",
    #        input_example=X.head(1),
    #        serialization_format="cloudpickle",
    #    )
        print("Modele logge dans MLflow")

        return {
            "message": "Entrainement termine",
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
