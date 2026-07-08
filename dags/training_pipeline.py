from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow import DAG
import subprocess
import sys


@task
def collect():
    subprocess.run([sys.executable, "../src/script.py"], check=True)

@task
def preprocess():
    subprocess.run([sys.executable, "../src/process.py"], check=True)

@task
def train():
    subprocess.run([sys.executable, "../src/train.py"], check=True)


@dag(
    dag_id='model_dag',
    tags=['ImmoMLOps', 'datascientest'],
    schedule_interval=None,
    start_date=datetime(2026, 7, 8), # day I wrote it
    catchup=False
    )

def model_dag():
    collect_task = collect()
    process_task = preprocess()
    train_task = train()

    collect_task >> process_task >> train_task

dag = model_dag()