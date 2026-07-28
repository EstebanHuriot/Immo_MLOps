FROM python:3.11-slim

WORKDIR /mlflow

COPY services/mlflow_tracking/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Le backend store et l'artifact store sont montes en volume par docker-compose
# (./mlruns:/mlflow/mlruns), donc le tracking survit aux redemarrages du conteneur.
CMD ["mlflow", "server", \
     "--backend-store-uri", "file:/mlflow/mlruns", \
     "--default-artifact-root", "file:/mlflow/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
