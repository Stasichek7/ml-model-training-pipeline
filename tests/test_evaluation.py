"""Verify metric definitions, score selection, and invalid evaluation data."""

import json

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.evaluation import evaluate_model


class HardPredictionModel:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, features):
        assert len(features) == len(self.predictions)
        return self.predictions


class ProbabilityModel(HardPredictionModel):
    classes_ = np.array([0, 1])

    def __init__(self, predictions, scores):
        super().__init__(predictions)
        self.scores = np.asarray(scores)

    def predict_proba(self, features):
        return np.column_stack([1 - self.scores, self.scores])


class DecisionModel(ProbabilityModel):
    def decision_function(self, features):
        return -self.scores


def test_perfect_predictions_produce_perfect_metrics_and_json_safe_results():
    y = np.array([0, 0, 1, 1])
    features = pd.DataFrame({"x": [1, 2, 3, 4]})
    result = evaluate_model(ProbabilityModel(y, [0.1, 0.2, 0.8, 0.9]), features, y)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert result[metric] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["classification_report"]["1"]["support"] == 2
    assert result["roc_curve"]["fpr"][0] == 0
    assert result["roc_curve"]["tpr"][-1] == 1
    json.dumps(result, allow_nan=False)


def test_imperfect_predictions_use_class_one_as_positive():
    features = pd.DataFrame({"x": [1, 2, 3, 4]})
    result = evaluate_model(
        ProbabilityModel([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9]), features, [0, 0, 1, 1]
    )
    for metric in ["accuracy", "precision", "recall", "f1"]:
        assert result[metric] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]


def test_decision_scores_take_precedence_over_probabilities():
    features = pd.DataFrame({"x": [1, 2, 3, 4]})
    result = evaluate_model(
        DecisionModel([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), features, [0, 0, 1, 1]
    )
    assert result["roc_auc"] == pytest.approx(1.0)


def test_estimator_without_scores_reports_roc_auc_as_unavailable():
    result = evaluate_model(HardPredictionModel([0, 1]), pd.DataFrame({"x": [1, 2]}), [0, 1])
    assert result["accuracy"] == 1
    assert result["roc_auc"] is None
    assert result["roc_curve"] is None


def test_single_class_evaluation_handles_undefined_metrics_without_nan():
    result = evaluate_model(
        ProbabilityModel([0, 0], [0.1, 0.2]), pd.DataFrame({"x": [1, 2]}), [0, 0]
    )
    assert result["accuracy"] == 1
    assert result["precision"] == 0
    assert result["recall"] == 0
    assert result["f1"] == 0
    assert result["roc_auc"] is None
    assert result["confusion_matrix"] == [[2, 0], [0, 0]]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("target", [[], [0], [[0], [1]]])
def test_evaluation_rejects_empty_mismatched_or_multidimensional_targets(target):
    with pytest.raises(ValueError, match="matching, nonempty"):
        evaluate_model(HardPredictionModel([0, 1]), pd.DataFrame({"x": [1, 2]}), target)


def test_evaluation_rejects_nonbinary_labels():
    with pytest.raises(ValueError, match="binary labels"):
        evaluate_model(HardPredictionModel([0, 2]), pd.DataFrame({"x": [1, 2]}), [0, 2])
