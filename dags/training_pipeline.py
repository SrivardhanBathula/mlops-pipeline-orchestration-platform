from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
import numpy as np


default_args = {
    "owner": "srivardhan",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "training_pipeline",
    default_args=default_args,
    description="Automated model training pipeline",
    schedule_interval="0 2 * * *",
    catchup=False,
    tags=["training", "production"],
)


def load_and_validate_data(**context):
    np.random.seed(42)
    n = 50000
    X = pd.DataFrame(np.random.randn(n, 20), columns=[f"feature_{i}" for i in range(20)])
    y = (X["feature_0"] + X["feature_1"] + np.random.randn(n) * 0.1 > 0).astype(int)
    X.to_parquet("/tmp/features.parquet")
    y.to_csv("/tmp/labels.csv", index=False)
    context["ti"].xcom_push(key="n_samples", value=n)


def train_model(**context):
    X = pd.read_parquet("/tmp/features.parquet")
    y = pd.read_csv("/tmp/labels.csv").squeeze()
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    with mlflow.start_run(run_name="automated_training"):
        model = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.05)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        f1 = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)
        mlflow.log_metrics({"f1_score": f1, "auc_roc": auc})
        mlflow.log_param("n_estimators", 200)
        mlflow.sklearn.log_model(model, "model", registered_model_name="production_classifier")
        run_id = mlflow.active_run().info.run_id
        context["ti"].xcom_push(key="run_id", value=run_id)
        context["ti"].xcom_push(key="f1_score", value=f1)


def validate_and_promote(**context):
    f1 = context["ti"].xcom_pull(key="f1_score")
    run_id = context["ti"].xcom_pull(key="run_id")
    if f1 >= 0.88:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage("production_classifier", 1, "Production")
        print(f"Model promoted to production. F1: {f1:.4f}")
    else:
        raise ValueError(f"Model validation failed. F1: {f1:.4f} < 0.88 threshold")


load_data = PythonOperator(task_id="load_and_validate_data", python_callable=load_and_validate_data, dag=dag)
train = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
validate = PythonOperator(task_id="validate_and_promote", python_callable=validate_and_promote, dag=dag)

load_data >> train >> validate
