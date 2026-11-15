"""Schema-checked inference using a trusted, already trained artifact."""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def load_artifact(path: str | Path) -> dict:
    """Load only artifacts you trust: joblib/pickle can execute arbitrary code."""
    artifact = joblib.load(Path(path))
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 1
        or not isinstance(artifact.get("pipeline"), Pipeline)
    ):
        raise ValueError("Unsupported or invalid model artifact")
    metadata = artifact.get("metadata", {})
    required = {"feature_names", "numeric_features", "target_names", "model_name"}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("Model artifact is missing schema metadata")
    names = metadata["feature_names"]
    numeric = metadata["numeric_features"]
    target_names = metadata["target_names"]
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(names) != len(set(names))
        or not isinstance(numeric, list)
        or not all(isinstance(name, str) and name in names for name in numeric)
        or not isinstance(target_names, list)
        or len(target_names) != 2
        or not all(isinstance(name, str) and name for name in target_names)
        or not isinstance(metadata["model_name"], str)
        or not metadata["model_name"]
    ):
        raise ValueError("Model artifact contains invalid schema metadata")
    return artifact


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # Python accepts JSON integers larger than a float can represent.
        return False


class PredictionService:
    """Load once and reuse a fitted pipeline across prediction requests."""

    def __init__(self, model_path: str | Path):
        self.artifact = load_artifact(model_path)
        self.pipeline = self.artifact["pipeline"]
        self.metadata = self.artifact["metadata"]

    def predict(self, instances: list[dict]) -> dict:
        if not isinstance(instances, list) or not 1 <= len(instances) <= 1000:
            raise ValueError("instances must contain between 1 and 1000 records")
        names = self.metadata["feature_names"]
        expected = set(names)
        numeric = set(self.metadata["numeric_features"])
        for index, row in enumerate(instances):
            if not isinstance(row, dict):
                raise ValueError(f"Record {index} must be an object")
            missing, extra = expected - row.keys(), row.keys() - expected
            if missing or extra:
                raise ValueError(
                    f"Record {index}: missing features {sorted(missing)}; "
                    f"unknown features {sorted(extra)}"
                )
            for name, value in row.items():
                if value is None:
                    continue
                if name in numeric:
                    if not _is_finite_number(value):
                        raise ValueError(f"Record {index}: {name} must be a finite number or null")
                elif not isinstance(value, (str, bool)):
                    raise ValueError(f"Record {index}: {name} must be a string, boolean, or null")
        frame = pd.DataFrame(instances, columns=names)
        # Explicit conversion also supports batches where a numeric column is all null.
        for name in numeric:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        predicted = self.pipeline.predict(frame).astype(int)
        probabilities = self.pipeline.predict_proba(frame)
        return {
            "model_name": self.metadata["model_name"],
            "classes": [int(label) for label in self.pipeline.classes_],
            "predictions": [
                {
                    "class_id": int(label),
                    "class_name": self.metadata["target_names"][int(label)],
                    "probabilities": np.asarray(probs).tolist(),
                }
                for label, probs in zip(predicted, probabilities, strict=True)
            ],
        }
