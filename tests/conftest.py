"""Small but real training run shared by integration tests."""

import pytest

from ml_pipeline import training
from ml_pipeline.data import load_dataset


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained")
    original = training.model_candidates

    def small_candidates(seed):
        candidates = original(seed)
        for name, (estimator, grid) in candidates.items():
            if name == "random_forest":
                estimator.set_params(n_estimators=12)
            if name == "gradient_boosting":
                grid = {**grid, "model__n_estimators": [10]}
            candidates[name] = (estimator, {key: [values[0]] for key, values in grid.items()})
        return candidates

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(training, "model_candidates", small_candidates)
        result = training.train_models(root, training.TrainingConfig(cv_folds=3))
    return root, result, load_dataset()
