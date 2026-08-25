import subprocess
import sys

from airflow.decorators import dag, task
from datetime import datetime


@dag(
    dag_id="model_dag",
    schedule=None,
    start_date=datetime(2026, 7, 8),
    catchup=False,
)
def training_pipeline():

    @task
    def collect():
        subprocess.run(
            [sys.executable, "-u", "/opt/airflow/src/collect.py"],
            check=True,
        )

    @task
    def preprocess():
        subprocess.run(
            [sys.executable, "-u", "/opt/airflow/src/process.py"],
            check=True,
        )

    @task
    def train():
        result = subprocess.run(
            [sys.executable, "-u", "/opt/airflow/src/train.py"],
            capture_output=True,
            text=True,
        )
    
        print("STDOUT:")
        print(result.stdout)
    
        print("STDERR:")
        print(result.stderr)
    
        if result.returncode != 0:
            raise RuntimeError(
                f"train.py failed with exit code {result.returncode}"
            )

    collect() >> preprocess() >> train()


training_pipeline()