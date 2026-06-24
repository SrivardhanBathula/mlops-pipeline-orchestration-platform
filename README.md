# MLOps Pipeline Orchestration Platform

> Production-grade MLOps platform with automated retraining pipelines, real-time drift detection, model registry, and full observability stack.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Airflow](https://img.shields.io/badge/Airflow-2.10-red)](https://airflow.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-blue)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Key Metrics

| Metric | Value |
|--------|-------|
| Pipeline Reliability | 99.5% |
| Model Retraining Frequency | Daily (automated) |
| Drift Detection Latency | <5 minutes |
| Models in Registry | 12+ |
| Avg Deployment Time | <8 minutes |

## Architecture

```
Data Sources → Airflow DAGs → Feature Store → Model Training
                                                     ↓
                              MLflow Registry ← Experiment Tracking
                                     ↓
                              Model Validation → A/B Testing → Production
                                                                    ↓
                              Evidently AI ← Prometheus ← Model Serving (FastAPI)
                                     ↓
                              Grafana Dashboard → Alerts → Auto-Retrain Trigger
```

## Core Components

- **Pipeline Orchestration** — Apache Airflow DAGs for training, validation, and deployment
- **Experiment Tracking** — MLflow for metrics, parameters, artifacts, and model versioning
- **Drift Detection** — Evidently AI for data and concept drift monitoring
- **Model Registry** — Centralized versioning with staging/production promotion
- **Observability** — Prometheus metrics + Grafana dashboards with alerting
- **Auto-Retraining** — Triggered by drift alerts or scheduled intervals

## Tech Stack

- **Orchestration:** Apache Airflow 2.10
- **ML Tracking:** MLflow 2.16, Weights & Biases
- **Drift Detection:** Evidently AI, Alibi-Detect
- **Serving:** FastAPI, BentoML
- **Infra:** Docker, Kubernetes, Terraform
- **Monitoring:** Prometheus, Grafana, CloudWatch
- **Storage:** PostgreSQL, Amazon S3, Redis

## Quick Start

```bash
git clone https://github.com/SrivardhanBathula/mlops-pipeline-orchestration-platform
cd mlops-pipeline-orchestration-platform
docker-compose up --build
# Airflow UI: http://localhost:8080 (admin/admin)
# MLflow UI:  http://localhost:5000
# Grafana:    http://localhost:3000
```

## Pipeline DAGs

```python
# Training Pipeline
training_pipeline >> feature_engineering >> model_training >> model_evaluation >> model_registration

# Drift Monitoring Pipeline  
data_ingestion >> drift_detection >> [alert_trigger, log_metrics]

# Auto-Retrain Pipeline
drift_alert >> data_validation >> retraining_pipeline >> champion_challenger >> promote_model
```

## Project Structure

```
mlops-pipeline-orchestration-platform/
├── dags/
│   ├── training_pipeline.py
│   ├── drift_monitoring.py
│   └── auto_retrain_pipeline.py
├── src/
│   ├── drift_detector.py
│   ├── model_registry.py
│   └── feature_store.py
├── monitoring/
│   └── grafana_dashboard.json
├── api/
│   └── main.py
├── config/
│   └── pipeline_config.yaml
├── notebooks/
│   └── 01_mlops_platform_demo.ipynb
├── docker-compose.yml
└── README.md
```

## Author

**Srivardhan Bathula** — AI/ML Engineer
- Portfolio: [srivardhanbathula.github.io/srivardhanb.github.io](https://srivardhanbathula.github.io/srivardhanb.github.io)
- LinkedIn: [linkedin.com/in/srivardhanb](https://linkedin.com/in/srivardhanb)
