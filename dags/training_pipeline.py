from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow import DAG
from airflow.utils.dates import days_ago
import subprocess
import sys


@task
def collect():
    subprocess.run([sys.executable, "../src/script.py"], check=True)





@dag(
    dag_id='model_dag',
    tags=['ImmoMLOps', 'datascientest'],
    schedule_interval=None,
    start_date=days_ago(0)
    )

def my_dag():
    my_task1 = collect()

my_dag = my_dag()