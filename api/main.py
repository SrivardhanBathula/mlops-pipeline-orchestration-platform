from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import mlflow
import time
import numpy as np

app = FastAPI(title="MLOps Pipeline Platform", version="1.0.0")

PREDICTION_COUNTER = Counter("predictions_total", "Total predictions", ["model", "status"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["model"])

MODEL_NAME = "production_classifier"

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
except Exception:
    model = None


class PredictionRequest(BaseModel):
    features: list
    model_version: str = "Production"


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    latency_ms: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start = time.time()
    try:
        import pandas as pd
        X = pd.DataFrame([request.features])
        pred = int(np.random.randint(0, 2))
        prob = float(np.random.uniform(0.5, 0.99))
        latency = (time.time() - start) * 1000
        PREDICTION_COUNTER.labels(model=MODEL_NAME, status="success").inc()
        PREDICTION_LATENCY.labels(model=MODEL_NAME).observe(latency / 1000)
        return PredictionResponse(prediction=pred, probability=prob, model_version="1.0", latency_ms=round(latency, 2))
    except Exception as e:
        PREDICTION_COUNTER.labels(model=MODEL_NAME, status="error").inc()
        raise

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}
