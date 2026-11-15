"""Load and validate reproducible binary-classification datasets."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.datasets import load_breast_cancer


@dataclass(frozen=True)
class DatasetBundle:
    """Features, binary labels and their human-readable class names."""

    X: pd.DataFrame
    y: pd.Series
    target_names: list[str]
    name: str


def validate_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """Reject malformed data while permitting missing feature values.

    Labels must be 0 and 1. Numeric and string/boolean categorical features
    are supported; missing values are handled inside the fitted pipeline.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError("Features must be a DataFrame and target must be a Series.")
    if X.empty:
        raise ValueError("The feature dataset must contain rows and columns.")
    if X.columns.duplicated().any():
        raise ValueError("Feature column names must be unique.")
    if any(not isinstance(column, str) or not column.strip() for column in X.columns):
        raise ValueError("Feature column names must be nonempty strings.")
    if "__target__" in X.columns:
        raise ValueError("Feature name '__target__' is reserved for saved holdout labels.")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("Features and target must have matching rows and indexes.")
    if y.isna().any():
        raise ValueError("Target labels cannot contain missing values.")
    if set(y.unique()) != {0, 1}:
        raise ValueError("Binary target labels must contain exactly classes 0 and 1.")

    for column in X.columns:
        values = X[column]
        if is_numeric_dtype(values.dtype) and not is_bool_dtype(values.dtype):
            if pd.api.types.is_complex_dtype(values.dtype):
                raise ValueError(f"Feature '{column}' cannot contain complex numbers.")
            if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"Feature '{column}' cannot contain infinite values.")
        else:
            observed = values.dropna()
            if not observed.map(lambda value: isinstance(value, (str, bool, np.bool_))).all():
                raise ValueError(
                    f"Categorical feature '{column}' must contain strings or booleans."
                )
            if observed.map(type).nunique() > 1:
                # sklearn's encoders require consistent types within a column.
                if not observed.map(lambda value: isinstance(value, (bool, np.bool_))).all():
                    raise ValueError(f"Categorical feature '{column}' has mixed value types.")


def load_dataset() -> DatasetBundle:
    """Load scikit-learn's bundled Wisconsin Diagnostic Breast Cancer data.

    No network access is required. Class 0 is malignant and class 1 benign.
    This educational benchmark is not a clinical diagnostic system.
    """
    source = load_breast_cancer(as_frame=True)
    X = source.data.rename(columns=lambda column: column.replace(" ", "_"))
    y = source.target.rename("target")
    validate_dataset(X, y)
    return DatasetBundle(
        X=X,
        y=y,
        target_names=list(source.target_names),
        name="breast_cancer",
    )


def load_csv_dataset(path: str | Path, target_column: str = "target") -> DatasetBundle:
    """Read a labeled CSV with a 0/1 target and numeric/categorical features."""
    # pandas renames duplicate/empty headers, so inspect the original schema first.
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or any(not name.strip() for name in header):
        raise ValueError("CSV column names must be nonempty strings.")
    if len(header) != len(set(header)):
        raise ValueError("CSV column names must be unique.")
    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column '{target_column}'.")
    y = frame[target_column].rename("target")
    X = frame.drop(columns=[target_column])
    validate_dataset(X, y)
    return DatasetBundle(X=X, y=y, target_names=["0", "1"], name=Path(path).stem)
