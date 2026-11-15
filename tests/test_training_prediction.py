"""Verify selection, split isolation, persisted inference, and command behavior."""

import json

import joblib
import numpy as np
import pytest

from ml_pipeline.cli import main
from ml_pipeline.prediction import PredictionService, load_artifact
from ml_pipeline.training import TrainingConfig


def test_training_selection_and_split_isolation(trained_run):
    root, result, dataset = trained_run
    metadata = result["metadata"]
    assert set(result["results"]) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "support_vector_machine",
    }
    assert result["best_model_name"] == max(
        result["results"],
        key=lambda name: result["results"][name]["cv_mean_roc_auc"],
    )
    assert not set(metadata["train_indices"]) & set(metadata["test_indices"])
    assert metadata["train_rows"] + metadata["test_rows"] == len(dataset.X)
    for values in result["results"].values():
        assert len(values["cv_fold_roc_auc"]) == 3
        assert values["cv_mean_roc_auc"] == pytest.approx(np.mean(values["cv_fold_roc_auc"]))
    fitted = load_artifact(root / "models/best_model.joblib")["pipeline"]
    scaler = fitted.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"]
    expected = dataset.X.loc[metadata["train_indices"]].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected)
    assert not np.allclose(scaler.mean_, dataset.X.mean().to_numpy())


def test_generated_report_artifacts(trained_run):
    root, _, _ = trained_run
    for name in [
        "evaluation.json",
        "model_comparison.csv",
        "confusion_matrix.png",
        "roc_curves.png",
        "eda_summary.json",
        "feature_distributions.png",
    ]:
        assert (root / "reports" / name).stat().st_size > 50


def test_serialization_and_prediction(trained_run, tmp_path):
    root, _, dataset = trained_run
    model_path = root / "models/best_model.joblib"
    artifact = load_artifact(model_path)
    copy_path = tmp_path / "copy.joblib"
    joblib.dump(artifact, copy_path)
    rows = dataset.X.head(3).to_dict(orient="records")
    before = PredictionService(model_path).predict(rows)
    after = PredictionService(copy_path).predict(rows)
    assert before == after
    assert len(before["predictions"]) == 3
    for prediction in before["predictions"]:
        assert sum(prediction["probabilities"]) == pytest.approx(1.0)
        assert prediction["class_name"] in {"malignant", "benign"}


@pytest.mark.parametrize("mutation", ["missing", "extra", "string", "infinity", "bool"])
def test_prediction_rejects_bad_schema(trained_run, mutation):
    root, _, dataset = trained_run
    row = dataset.X.iloc[0].to_dict()
    key = next(iter(row))
    if mutation == "missing":
        row.pop(key)
    elif mutation == "extra":
        row["unknown"] = 1
    else:
        row[key] = {"string": "3.2", "infinity": float("inf"), "bool": True}[mutation]
    with pytest.raises(ValueError):
        PredictionService(root / "models/best_model.joblib").predict([row])


def test_prediction_accepts_nulls_and_feature_reordering(trained_run):
    root, _, dataset = trained_run
    service = PredictionService(root / "models/best_model.joblib")
    row = dataset.X.iloc[0].to_dict()
    row[next(iter(row))] = None
    assert service.predict([row]) == service.predict([dict(reversed(list(row.items())))])


@pytest.mark.parametrize("rows", [[], [{}] * 1001, [42]])
def test_prediction_rejects_invalid_batches(trained_run, rows):
    root, _, _ = trained_run
    with pytest.raises(ValueError):
        PredictionService(root / "models/best_model.joblib").predict(rows)


def test_rejects_invalid_artifact(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"schema_version": 99}, path)
    with pytest.raises(ValueError, match="artifact"):
        load_artifact(path)


@pytest.mark.parametrize("kwargs", [{"cv_folds": 1}, {"test_size": 0.9}, {"n_jobs": 0}])
def test_training_config_validation(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_cli_evaluation_and_checksum(trained_run, tmp_path, capsys):
    root, _, _ = trained_run
    args = [
        "evaluate",
        "--model",
        str(root / "models/best_model.joblib"),
        "--data",
        str(root / "models/holdout.csv"),
        "--output",
        str(tmp_path / "metrics.json"),
    ]
    assert main(args) == 0
    assert json.loads((tmp_path / "metrics.json").read_text())["roc_auc"] > 0.9
    bad = tmp_path / "changed.csv"
    bad.write_text("changed", encoding="utf-8")
    args[4] = str(bad)
    assert main(args) == 1
    assert "checksum" in capsys.readouterr().err.lower()


def test_cli_prediction_and_report(trained_run, tmp_path):
    root, _, dataset = trained_run
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"instances": dataset.X.head(1).to_dict(orient="records")}))
    output = tmp_path / "predictions.json"
    assert (
        main(
            [
                "predict",
                "--input",
                str(path),
                "--model",
                str(root / "models/best_model.joblib"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(json.loads(output.read_text())["predictions"]) == 1
    assert main(["report", "--reports-dir", str(root / "reports")]) == 0


def test_cli_missing_model_returns_error(tmp_path):
    assert main(["predict", "--input", str(tmp_path / "missing.json")]) == 1
