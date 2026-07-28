# DVC + DagsHub — commandes à lancer ce soir

## 0. Dépendances

```bash
pip install -r requirements.txt
```

## 1. Créer le repo sur DagsHub

1. Va sur https://dagshub.com, connecte-toi, clique "Create" -> "Connect a repo".
2. Connecte ton repo GitHub `Immo_MLOps` existant.
3. Onglet **Remote** : copie les commandes exactes affichées avec TES identifiants.

## 2. Initialiser DVC

```bash
dvc init
```

## 3. Ajouter les données et le modèle

```bash
dvc add data/annonces_france/df_france_ventes_cleaned.csv
dvc add models/best_model.pkl
dvc add models/feature_info.pkl

git add data/annonces_france/df_france_ventes_cleaned.csv.dvc \
        models/best_model.pkl.dvc models/feature_info.pkl.dvc \
        .gitignore .dvc/config
git commit -m "Track dataset and model with DVC"
```

## 4. Connecter le remote DVC à DagsHub

```bash
dvc remote add origin https://dagshub.com/<user>/<repo>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user>
dvc remote modify origin --local password <token>
```

## 5. Push / pull

```bash
dvc push
git push
```

## 6. Centraliser le tracking MLflow sur DagsHub

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
export MLFLOW_TRACKING_USERNAME="<user>"
export MLFLOW_TRACKING_PASSWORD="<token>"

python src/train.py
```

`src/train.py` lit ces 3 variables d'environnement (sinon tracking local par
fichiers, `mlruns/`).

## 7. Vérifier

- DagsHub, onglet **Experiments** : les runs MLflow doivent apparaître.
- DagsHub, onglet **Files** : les fichiers DVC doivent apparaître en taille réelle.

## À dire à l'oral

- DVC : versionner des gros fichiers (data, modèles) sans alourdir git.
- DagsHub : centraliser tracking MLflow + stockage DVC en un seul endroit
  partagé par toute l'équipe.
