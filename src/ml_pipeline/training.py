"""Tune pipelines on training folds and evaluate once on a reserved holdout."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ml_pipeline.data import DatasetBundle, load_dataset, validate_dataset
from ml_pipeline.evaluation import evaluate_model
from ml_pipeline.preprocessing import build_preprocessor
from ml_pipeline.reporting import generate_eda, generate_evaluation_report


@dataclass(frozen=True)
class TrainingConfig:
    """Small, reproducible defaults suitable for a local CPU."""

    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = 1

    def __post_init__(self):
        if not 0.1 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.1 and 0.5")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


def model_candidates(seed: int) -> dict:
    """Return independent estimators and intentionally bounded search spaces."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=seed),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=1),
            {"model__max_depth": [None, 6], "model__min_samples_leaf": [1, 3]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {"model__n_estimators": [60, 100], "model__learning_rate": [0.05, 0.1]},
        ),
        "support_vector_machine": (
            SVC(probability=True, random_state=seed),
            {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.01]},
        ),
    }


def train_models(
    output_dir: str | Path = ".",
    config: TrainingConfig | None = None,
    dataset: DatasetBundle | None = None,
) -> dict:
    """Train all candidates, select by CV ROC-AUC, persist the winning pipeline.

    Ties keep candidate declaration order. The test partition is never passed to
    fitting, tuning, EDA, or selection. CV values are tuning scores, not unbiased
    estimates of generalization after model selection.
    """
    config = config or TrainingConfig()
    dataset = dataset or load_dataset()
    validate_dataset(dataset.X, dataset.y)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=dataset.y,
    )
    if y_train.value_counts().min() < config.cv_folds:
        raise ValueError("Each training class must have at least cv_folds samples")
    root = Path(output_dir)
    models_dir, reports_dir = root / "models", root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    generate_eda(X_train, y_train, reports_dir)
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    searches = {}
    results = {}
    for name, (estimator, grid) in model_candidates(config.random_state).items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=config.n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        searches[name] = search
        idx = search.best_index_
        results[name] = {
            "cv_mean_roc_auc": float(search.best_score_),
            "cv_std_roc_auc": float(search.cv_results_["std_test_score"][idx]),
            "best_params": search.best_params_,
            "cv_fold_roc_auc": [
                float(search.cv_results_[f"split{i}_test_score"][idx])
                for i in range(config.cv_folds)
            ],
        }
        pd.DataFrame(search.cv_results_).to_csv(reports_dir / f"cv_{name}.csv", index=False)
    best_name = max(results, key=lambda name: results[name]["cv_mean_roc_auc"])
    # Selection is final before any holdout metric is calculated.
    for name, search in searches.items():
        results[name].update(evaluate_model(search.best_estimator_, X_test, y_test))
    holdout_path = models_dir / "holdout.csv"
    holdout = X_test.copy()
    holdout["__target__"] = y_test
    holdout.to_csv(holdout_path, index=False)
    metadata = {
        "model_name": best_name,
        "dataset": dataset.name,
        "created_at": datetime.now(UTC).isoformat(),
        "feature_names": dataset.X.columns.tolist(),
        "numeric_features": dataset.X.select_dtypes(include="number").columns.tolist(),
        "target_names": dataset.target_names,
        "positive_class": 1,
        "selection_metric": "mean training CV ROC-AUC",
        "cv_mean_roc_auc": results[best_name]["cv_mean_roc_auc"],
        "cv_std_roc_auc": results[best_name]["cv_std_roc_auc"],
        "best_params": results[best_name]["best_params"],
        "config": asdict(config),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_indices": X_train.index.tolist(),
        "test_indices": X_test.index.tolist(),
        "holdout_sha256": hashlib.sha256(holdout_path.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    artifact = {
        "schema_version": 1,
        "pipeline": searches[best_name].best_estimator_,
        "metadata": metadata,
    }
    temporary = models_dir / "best_model.joblib.tmp"
    joblib.dump(artifact, temporary)
    temporary.replace(models_dir / "best_model.joblib")
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    generate_evaluation_report(results, reports_dir, best_name)
    return {"best_model_name": best_name, "results": results, "metadata": metadata}
