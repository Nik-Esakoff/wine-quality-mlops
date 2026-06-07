import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/airflow/dags/wine-quality-mlops")


with DAG(
    dag_id="wine_quality_training",
    description="Daily training pipeline for wine quality model",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "wine"],
) as dag:
    load_data = BashOperator(
        task_id="load_data",
        bash_command=f"cd '{PROJECT_DIR}' && dvc pull",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd '{PROJECT_DIR}' && python src/train.py",
    )

    save_model = BashOperator(
        task_id="save_model",
        bash_command=f"cd '{PROJECT_DIR}' && dvc add models && dvc push",
    )

    load_data >> train_model >> save_model
