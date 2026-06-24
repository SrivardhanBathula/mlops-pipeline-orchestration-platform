import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class FeatureStore:
    """Lightweight in-memory feature store with versioning and lineage tracking."""

    def __init__(self, storage_path: str = "/tmp/feature_store"):
        self.storage_path = storage_path
        self._feature_groups: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, Dict] = {}

    def create_feature_group(
        self,
        name: str,
        features: pd.DataFrame,
        entity_col: str = "entity_id",
        description: str = ""
    ) -> str:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"{name}:{version}"
        self._feature_groups[key] = features.copy()
        self._metadata[key] = {
            "name": name,
            "version": version,
            "entity_col": entity_col,
            "description": description,
            "n_rows": len(features),
            "n_features": len(features.columns),
            "created_at": datetime.now().isoformat(),
            "checksum": hashlib.md5(features.to_json().encode()).hexdigest()
        }
        logger.info(f"Feature group '{name}' v{version} created: {features.shape}")
        return version

    def get_features(
        self,
        name: str,
        entity_ids: Optional[List] = None,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        if version:
            key = f"{name}:{version}"
        else:
            keys = [k for k in self._feature_groups if k.startswith(f"{name}:")]
            if not keys:
                raise KeyError(f"Feature group '{name}' not found")
            key = sorted(keys)[-1]

        df = self._feature_groups[key]
        meta = self._metadata[key]

        if entity_ids:
            entity_col = meta["entity_col"]
            df = df[df[entity_col].isin(entity_ids)]

        return df

    def list_feature_groups(self) -> List[Dict[str, Any]]:
        return [v for v in self._metadata.values()]
