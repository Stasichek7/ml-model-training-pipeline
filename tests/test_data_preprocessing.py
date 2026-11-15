"""Dataset integrity and preprocessing behavior independent of model choice."""

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_pipeline.data import load_csv_dataset, load_dataset, validate_dataset
from ml_pipeline.preprocessing import build_preprocessor


def test_builtin_dataset_has_expected_schema_and_labels():
    dataset = load_dataset()
    assert dataset.X.shape == (569, 30)
    assert dataset.y.value_counts().to_dict() == {1: 357, 0: 212}
    assert dataset.target_names == ["malignant", "benign"]
    assert dataset.name == "breast_cancer"
    assert dataset.X.index.equals(dataset.y.index)
    assert all(" " not in name for name in dataset.X.columns)
    assert dataset.X.columns.is_unique


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (pd.DataFrame(), pd.Series(dtype=int), "contain rows"),
        (pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"]), pd.Series([0, 1]), "unique"),
        (pd.DataFrame({0: [1, 2]}), pd.Series([0, 1]), "nonempty strings"),
        (pd.DataFrame({"x": [1, 2]}), pd.Series([0]), "matching"),
        (pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1], index=[1, 0]), "matching"),
        (pd.DataFrame({"x": [1, 2]}), pd.Series([0, np.nan]), "missing"),
        (pd.DataFrame({"x": [1, 2]}), pd.Series([1, 1]), "exactly classes"),
        (pd.DataFrame({"x": [1, 2]}), pd.Series([0, 2]), "exactly classes"),
        (pd.DataFrame({"x": [1, np.inf]}), pd.Series([0, 1]), "infinite"),
        (pd.DataFrame({"x": [1 + 2j, 3 + 4j]}), pd.Series([0, 1]), "complex"),
        (pd.DataFrame({"x": [[1], [2]]}), pd.Series([0, 1]), "strings or booleans"),
        (pd.DataFrame({"x": ["a", True]}), pd.Series([0, 1]), "mixed value types"),
    ],
)
def test_dataset_validation_rejects_malformed_input(X, y, message):
    with pytest.raises(ValueError, match=message):
        validate_dataset(X, y)


def test_dataset_validation_allows_missing_features_and_boolean_categories():
    features = pd.DataFrame(
        {
            "amount": [1.0, np.nan, 3.0],
            "category": ["a", None, "b"],
            "enabled": [True, False, True],
        }
    )
    validate_dataset(features, pd.Series([0, 1, 1]))


def test_csv_loading_preserves_target_and_categorical_features(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"amount": [1.0, 2.0], "group": ["a", "b"], "label": [0, 1]}).to_csv(
        path, index=False
    )
    dataset = load_csv_dataset(path, target_column="label")
    assert list(dataset.X.columns) == ["amount", "group"]
    assert dataset.y.tolist() == [0, 1]
    assert dataset.name == "sample"


def test_csv_loading_rejects_absent_target(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing target"):
        load_csv_dataset(path)


def test_preprocessing_handles_missing_values_and_unknown_categories():
    training = pd.DataFrame(
        {
            "amount": [0.0, 2.0, np.nan],
            "group": ["a", "b", None],
            "empty": [np.nan, np.nan, np.nan],
            "empty_category": pd.Series([None, None, None], dtype=object),
        }
    )
    preprocessor = build_preprocessor(training)
    transformed = preprocessor.fit_transform(training)
    unseen = pd.DataFrame(
        {
            "amount": [100.0],
            "group": ["unseen"],
            "empty": [np.nan],
            "empty_category": [None],
        }
    )
    predicted_features = preprocessor.transform(unseen)
    assert np.isfinite(transformed).all()
    assert np.isfinite(predicted_features).all()
    assert predicted_features.shape[1] == transformed.shape[1]
    assert preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_[0] == 1
    assert preprocessor.named_transformers_["numeric"].named_steps["scaler"].mean_[0] == 1


def test_preprocessing_is_serializable_and_does_not_fit_on_inference_data(tmp_path):
    training = pd.DataFrame({"amount": [0.0, 2.0], "group": ["a", "b"]})
    preprocessor = build_preprocessor(training).fit(training)
    unseen = pd.DataFrame({"amount": [100.0], "group": ["new"]})
    expected = preprocessor.transform(unseen)
    assert expected[0, 0] == pytest.approx(99.0)
    np.testing.assert_array_equal(expected[0, 1:], [0.0, 0.0])
    path = tmp_path / "preprocessor.joblib"
    joblib.dump(preprocessor, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.transform(unseen), expected)


@pytest.mark.parametrize("features", [pd.DataFrame(), [[1], [2]]])
def test_preprocessing_rejects_empty_or_non_dataframe_features(features):
    with pytest.raises(ValueError, match="nonempty feature DataFrame"):
        build_preprocessor(features)
