"""Binary classification metrics computed on an untouched evaluation split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: Any) -> dict[str, Any]:
    """Return JSON-safe metrics, using encoded class 1 as the positive class.

    ROC-AUC uses decision scores when provided and otherwise probabilities. It
    is unavailable when the estimator has neither interface or the evaluation
    data contains only one class. Other metrics still describe that split.
    """
    y_true = np.asarray(y_test)
    if y_true.ndim != 1 or len(y_true) == 0 or len(X_test) != len(y_true):
        raise ValueError("Evaluation requires matching, nonempty X and one-dimensional y.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Evaluation targets must use binary labels 0 and 1.")

    predictions = model.predict(X_test)
    scores = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            scores = model.predict_proba(X_test)[:, classes.index(1)]
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, predictions, labels=[0, 1], output_dict=True, zero_division=0
        ),
        "roc_curve": None,
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
        result["roc_curve"] = {
            "fpr": false_positive_rate.tolist(),
            "tpr": true_positive_rate.tolist(),
        }
    return result
