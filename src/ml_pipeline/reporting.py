"""Headless exploratory analysis and reproducible evaluation reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Keep the font cache writable in restricted Windows profiles.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def generate_eda(X: pd.DataFrame, y: pd.Series, output_dir: Path | str) -> dict[str, str]:
    """Write descriptive statistics and plots for the supplied training split.

    The caller controls the split so held-out observations need not influence
    exploratory decisions. Missing values are counted before preprocessing.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if X.empty or len(X) != len(y):
        raise ValueError("EDA requires matching, nonempty features and targets.")
    numeric = X.select_dtypes(include="number")
    class_counts = y.value_counts().sort_index()
    missing = pd.DataFrame(
        {
            "feature": X.columns,
            "missing_count": X.isna().sum().to_numpy(),
            "missing_fraction": X.isna().mean().to_numpy(),
        }
    )
    missing.to_csv(output / "missing_values.csv", index=False)
    summary = {
        "rows": len(X),
        "features": len(X.columns),
        "numeric_features": list(numeric.columns),
        "categorical_features": [column for column in X.columns if column not in numeric.columns],
        "class_counts": class_counts.to_dict(),
        "missing_cells": int(X.isna().sum().sum()),
        "numeric_statistics": numeric.describe().to_dict() if len(numeric.columns) else {},
    }
    _write_json(output / "eda_summary.json", summary)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(label) for label in class_counts.index], class_counts.to_numpy(), color="#2563eb")
    ax.set(
        xlabel="Encoded target class", ylabel="Observations", title="Training class distribution"
    )
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)

    selected_columns = list(numeric.columns[:6]) or list(X.columns[:6])
    rows = (len(selected_columns) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.3 * rows), squeeze=False)
    for ax, column in zip(axes.flat, selected_columns, strict=False):
        observed = X[column].dropna()
        if column in numeric.columns:
            ax.hist(observed, bins=22, color="#0f766e", alpha=0.85)
            ax.set_ylabel("Observations")
        else:
            counts = observed.astype(str).value_counts().head(10)
            ax.bar(counts.index, counts.to_numpy(), color="#0f766e")
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(str(column).replace("_", " "))
    for ax in list(axes.flat)[len(selected_columns) :]:
        ax.set_visible(False)
    fig.suptitle("Training feature distributions (first six features)")
    fig.tight_layout()
    fig.savefig(output / "feature_distributions.png", dpi=160)
    plt.close(fig)
    return {
        name: str(output / name)
        for name in (
            "eda_summary.json",
            "missing_values.csv",
            "class_distribution.png",
            "feature_distributions.png",
        )
    }


def generate_evaluation_report(
    results: dict[str, dict[str, Any]],
    output_dir: Path | str,
    best_model_name: str,
) -> dict[str, str]:
    """Write machine-readable results and comparison, confusion, and ROC plots."""
    if not results or best_model_name not in results:
        raise ValueError("The selected model must be present in nonempty evaluation results.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "evaluation.json",
        {
            "best_model_name": best_model_name,
            "selection_metric": "mean stratified training cross-validation ROC-AUC",
            "positive_class": 1,
            "results": results,
        },
    )
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "cv_mean_roc_auc",
        "cv_std_roc_auc",
    ]
    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "selected": name == best_model_name,
                **{metric: values.get(metric) for metric in metrics},
            }
            for name, values in results.items()
        ]
    )
    comparison = comparison.sort_values("cv_mean_roc_auc", ascending=False, na_position="last")
    comparison.to_csv(output / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(comparison))
    width = 0.25
    for offset, (metric, label, color) in enumerate(
        [
            ("cv_mean_roc_auc", "Training CV ROC-AUC", "#2563eb"),
            ("roc_auc", "Test ROC-AUC", "#0f766e"),
            ("f1", "Test F1", "#d97706"),
        ]
    ):
        values = pd.to_numeric(comparison[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(positions + (offset - 1) * width, values, width, label=label, color=color)
    ax.set_xticks(positions, comparison["model"], rotation=15, ha="right")
    ax.set(
        ylim=(0, 1.05), ylabel="Score", title="Model comparison (selection uses training CV only)"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)

    matrix = np.asarray(results[best_model_name]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    rendered = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(rendered, ax=ax)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted class",
        ylabel="Actual class",
        title=f"Selected model: {best_model_name}",
    )
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, values in results.items():
        curve = values.get("roc_curve")
        if curve:
            ax.plot(curve["fpr"], curve["tpr"], label=f"{name} (AUC {values['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Held-out test ROC curves (positive class = 1)",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "roc_curves.png", dpi=160)
    plt.close(fig)
    return {
        name: str(output / name)
        for name in (
            "evaluation.json",
            "model_comparison.csv",
            "model_comparison.png",
            "confusion_matrix.png",
            "roc_curves.png",
        )
    }


def render_existing_report(output_dir: Path | str) -> dict[str, str]:
    """Regenerate visualizations from an existing evaluation.json file."""
    output = Path(output_dir)
    payload = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    return generate_evaluation_report(payload["results"], output, payload["best_model_name"])
