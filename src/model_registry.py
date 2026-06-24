import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(self, run_id: str, model_name: str, artifact_path: str = "model") -> str:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered {model_name} v{mv.version}")
        return mv.version

    def promote_to_staging(self, model_name: str, version: str):
        self.client.transition_model_version_stage(model_name, version, "Staging")
        logger.info(f"Promoted {model_name} v{version} to Staging")

    def promote_to_production(self, model_name: str, version: str):
        prod_versions = self.client.get_latest_versions(model_name, stages=["Production"])
        for v in prod_versions:
            self.client.transition_model_version_stage(model_name, v.version, "Archived")
        self.client.transition_model_version_stage(model_name, version, "Production")
        logger.info(f"Promoted {model_name} v{version} to Production")

    def get_production_model(self, model_name: str):
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise ValueError(f"No production model found for {model_name}")
        return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

    def list_models(self) -> List[Dict]:
        return [
            {"name": m.name, "latest_version": m.latest_versions[-1].version if m.latest_versions else None}
            for m in self.client.search_registered_models()
        ]
