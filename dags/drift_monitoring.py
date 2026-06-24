from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

default_args = {
    "owner": "srivardhan",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    "drift_monitoring_pipeline",
    default_args=default_args,
    description="Real-time data and model drift monitoring",
    schedule_interval="*/15 * * * *",
    catchup=False,
    tags=["monitoring", "drift", "production"],
)


def load_reference_data(**context):
    np.random.seed(42)
    n = 5000
    ref = pd.DataFrame(np.random.randn(n, 20), columns=[f"f_{i}" for i in range(20)])
    ref.to_parquet("/tmp/reference_data.parquet")
    logger.info(f"Reference data loaded: {ref.shape}")


def load_current_data(**context):
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    n = 1000
    drift_factor = np.random.choice([1.0, 2.5], p=[0.85, 0.15])
    curr = pd.DataFrame(
        np.random.randn(n, 20) * drift_factor + (drift_factor - 1),
        columns=[f"f_{i}" for i in range(20)]
    )
    curr.to_parquet("/tmp/current_data.parquet")
    logger.info(f"Current data loaded: {curr.shape}, drift_factor={drift_factor}")


def detect_drift(**context):
    from src.drift_detector import DriftDetector
    ref = pd.read_parquet("/tmp/reference_data.parquet")
    curr = pd.read_parquet("/tmp/current_data.parquet")
    detector = DriftDetector(drift_threshold=0.15)
    report = detector.detect_drift(ref, curr)
    context["ti"].xcom_push(key="drift_report", value=report)
    logger.info(f"Drift detection complete: {report}")
    return report


def check_drift_threshold(**context):
    report = context["ti"].xcom_pull(key="drift_report")
    if report and report.get("drift_detected"):
        return "trigger_retraining"
    return "log_healthy_status"


def trigger_retraining(**context):
    logger.warning("DRIFT DETECTED — triggering automated retraining pipeline")


def log_healthy_status(**context):
    logger.info("No drift detected — model serving healthy")


load_ref = PythonOperator(task_id="load_reference_data", python_callable=load_reference_data, dag=dag)
load_curr = PythonOperator(task_id="load_current_data", python_callable=load_current_data, dag=dag)
detect = PythonOperator(task_id="detect_drift", python_callable=detect_drift, dag=dag)
branch = BranchPythonOperator(task_id="check_drift_threshold", python_callable=check_drift_threshold, dag=dag)
retrain = PythonOperator(task_id="trigger_retraining", python_callable=trigger_retraining, dag=dag)
healthy = PythonOperator(task_id="log_healthy_status", python_callable=log_healthy_status, dag=dag)

[load_ref, load_curr] >> detect >> branch >> [retrain, healthy]
