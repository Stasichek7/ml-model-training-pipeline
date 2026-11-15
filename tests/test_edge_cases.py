"""Regression coverage for malformed numeric inputs and persisted schemas."""

import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml_pipeline.api import create_app
from ml_pipeline.data import load_csv_dataset, validate_dataset
from ml_pipeline.prediction import PredictionService, load_artifact


def test_prediction_rejects_integer_too_large_for_float(trained_run):
    root, _, dataset = trained_run
    row = dataset.X.iloc[0].to_dict()
    row["mean_radius"] = 10**400
    with pytest.raises(ValueError, match="finite number"):
        PredictionService(root / "models/best_model.joblib").predict([row])


def test_api_rejects_integer_too_large_for_float(trained_run):
    root, _, dataset = trained_run
    row = dataset.X.iloc[0].to_dict()
    row["mean_radius"] = 10**400
    with TestClient(create_app(root / "models/best_model.joblib")) as client:
        response = client.post("/predict", json={"instances": [row]})
    assert response.status_code == 422


@pytest.mark.parametrize("literal", ["1e999", "Infinity"])
def test_api_returns_json_error_for_nonfinite_numeric_literal(trained_run, literal):
    root, _, dataset = trained_run
    row = dataset.X.iloc[0].to_dict()
    row["mean_radius"] = "__nonfinite__"
    payload = json.dumps({"instances": [row]}).replace('"__nonfinite__"', literal)
    with TestClient(
        create_app(root / "models/best_model.joblib"),
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/predict", content=payload, headers={"content-type": "application/json"}
        )
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize("metadata", [None, [], "invalid", {"feature_names": ["x"]}])
def test_artifact_rejects_malformed_metadata(trained_run, tmp_path, metadata):
    root, _, _ = trained_run
    artifact = load_artifact(root / "models/best_model.joblib")
    artifact["metadata"] = metadata
    invalid = tmp_path / "invalid-metadata.joblib"
    joblib.dump(artifact, invalid)
    with pytest.raises(ValueError, match="schema metadata"):
        load_artifact(invalid)


def test_dataset_rejects_reserved_holdout_target_feature():
    features = pd.DataFrame({"__target__": [1.0, 2.0], "x": [3.0, 4.0]})
    target = pd.Series([0, 1])
    with pytest.raises(ValueError, match="reserved"):
        validate_dataset(features, target)


@pytest.mark.parametrize(
    ("header", "message"),
    [("x,x,target", "unique"), ("x,,target", "nonempty"), ("x, ,target", "nonempty")],
)
def test_csv_rejects_duplicate_or_blank_original_headers(tmp_path, header, message):
    path = tmp_path / "invalid-headers.csv"
    path.write_text(f"{header}\n1,2,0\n3,4,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_csv_dataset(path)
