# Plan complet — de maintenant à la soutenance

## Étape 1 — Vérifier que rien n'est cassé [critique, ~10 min]

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

Tu dois voir **9 passed**.

## Étape 2 — Créer le compte DagsHub [important, ~10 min]

Voir `DVC_DAGSHUB_SETUP.md`. Fais-le avant d'entraîner pour logger
directement vers DagsHub.

## Étape 3 — Générer les données et entraîner [critique, 30–90 min]

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
export MLFLOW_TRACKING_USERNAME="<user>"
export MLFLOW_TRACKING_PASSWORD="<token>"

python src/collect.py
python src/process.py
python src/train.py
```

Note bien RMSE / MAE / R² affichés à la fin — pour le slide 7.
Si trop long : `export N_SAMPLE=20000` avant `train.py`.

## Étape 4 — DVC [important, ~10 min]

Voir `DVC_DAGSHUB_SETUP.md` étapes 2 à 5.

## Étape 5 — Push GitHub (déclenche la CI) [critique, ~5 min]

```bash
git add .
git commit -m "Add mlflow-tracking service, DVC, CI, tests, nginx bonus"
git push
```

Vérifie l'onglet **Actions** sur GitHub.

## Étape 6 — Stack complète en local [critique, ~10 min]

```bash
docker compose up --build
```

- Airflow : http://localhost:8081 (admin/admin)
- Prediction API : http://localhost:8000/docs
- Training API : http://localhost:8001/docs
- MLflow local : http://localhost:5000

## Étape 7 — Bonus nginx [bonus]

```bash
docker compose --profile bonus up --build
```

## Étape 8 — Compléter les slides [critique, ~10 min]

Remplace les `[À COMPLETER]` du slide 7 (RMSE/MAE/R²) dans
`Immo_MLOps_soutenance.pptx`.

## Étape 9 — Répéter [important]

Les notes de présentateur sont dans le `.pptx`, une par slide.

## Si tu dois couper quelque chose

1. Nginx (bonus explicite dans les consignes).
2. DVC/DagsHub — assume-le à l'oral si non fait.
3. CI GitHub Actions — avoir `pytest tests/ -v` qui passe en local suffit
   si la CI n'a pas le temps d'être vérifiée.

Ne saute jamais l'étape 1 (vérifier le code) ni l'étape 8 (vrais chiffres
dans les slides).
