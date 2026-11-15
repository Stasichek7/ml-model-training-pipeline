"""Build preprocessing that is fitted only inside training/CV pipelines."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _normalize_categorical_missing(values: pd.DataFrame) -> pd.DataFrame:
    """Convert None/pd.NA to np.nan so the categorical imputer handles both."""
    frame = pd.DataFrame(values).astype(object)
    return frame.where(frame.notna(), np.nan)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create unfitted numeric and categorical transformations from the schema.

    Empty columns are retained and unseen categories are safely ignored.
    The caller must put this transformer inside the model's Pipeline before CV.
    """
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("Preprocessing requires a nonempty feature DataFrame.")
    numeric_columns = [
        column
        for column in X.columns
        if is_numeric_dtype(X[column].dtype) and not is_bool_dtype(X[column].dtype)
    ]
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "normalize_missing",
                FunctionTransformer(_normalize_categorical_missing, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
