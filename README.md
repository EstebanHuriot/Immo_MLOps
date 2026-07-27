# Immo_MLOps

Machine learning project for real estate price prediction in France.</br>
The goal of this project is to build a simple MLOps pipeline that collects real estate data, preprocesses it, trains a model, evaluates its performance, and serves predictions through an API.


## Project structure

The project contains:
- An airflow pipeline for data collection, preprocessing and model training.
- A dedicated API for prediction.
- A dedicated API to trigger training.
- MLflow for experience tracking.
- DockerCompose for orchestration.
- PostegreSQL as an airflow metadata base


```text
Immo_MLOps/
├── dags/
│   └── training_pipeline.py
│
├── services/
│   ├── prediction_api/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── training_api/
│       ├── app.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── src/
│   ├── collect.py
│   ├── process.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── data/
├── models/
├── mlruns/
├── logs/
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Model

The model currently used is a RandomForestRegressor.
