import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesSummaryMetric
from typing import Dict, Any, Optional
import mlflow
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold
        self.report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesSummaryMetric(),
        ])

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        column_mapping: Optional[Dict] = None
    ) -> Dict[str, Any]:
        self.report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        results = self.report.as_dict()
        drift_detected = False
        drifted_features = []

        for metric in results.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                drift_share = metric["result"].get("drift_share", 0)
                drift_detected = drift_share > self.drift_threshold
                drifted_features = metric["result"].get("number_of_drifted_columns", 0)
                break

        drift_report = {
            "drift_detected": drift_detected,
            "drift_share": drift_share if 'drift_share' in dir() else 0,
            "drifted_features_count": drifted_features,
            "threshold": self.drift_threshold,
            "action_required": drift_detected
        }

        with mlflow.start_run(run_name="drift_monitoring"):
            mlflow.log_metrics({
                "drift_share": drift_report["drift_share"],
                "drifted_features": drift_report["drifted_features_count"]
            })
            mlflow.log_dict(drift_report, "drift_report.json")

        if drift_detected:
            logger.warning(f"DRIFT DETECTED: {drift_share:.2%} of features drifted. Triggering retraining.")

        return drift_report
