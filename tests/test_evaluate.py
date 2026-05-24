"""
tests/test_evaluate.py
----------------------
Unit tests for billionaires.models.evaluate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from billionaires.models.evaluate import (
    evaluate_classifier,
    evaluate_clusters,
    evaluate_regressor,
    metrics_table,
)


def test_evaluate_classifier_returns_core_metrics() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.45, 0.2, 0.8, 0.7])

    metrics = evaluate_classifier(y_true, y_pred, y_prob, split="unit")

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert "Self-Made" in metrics["classification_report"]


def test_evaluate_regressor_with_raw_scale_metrics() -> None:
    y_true = np.log1p(np.array([1.0, 2.0, 4.0, 8.0]))
    y_pred = np.log1p(np.array([1.2, 1.8, 5.0, 7.0]))

    metrics = evaluate_regressor(y_true, y_pred, exponentiated=True, split="unit")

    assert set(metrics) == {"mae", "rmse", "r2", "mae_raw", "rmse_raw", "mape_raw"}
    assert metrics["mae"] >= 0.0
    assert metrics["rmse_raw"] >= metrics["mae_raw"] - 1e-9
    assert metrics["mape_raw"] >= 0.0


def test_evaluate_clusters_handles_single_cluster() -> None:
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    labels = np.zeros(len(X), dtype=int)

    metrics = evaluate_clusters(X, labels)

    assert np.isnan(metrics["silhouette"])
    assert np.isnan(metrics["davies_bouldin"])
    assert metrics["n_clusters"] == 1


def test_evaluate_clusters_returns_scores_for_multiple_clusters() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [4.0, 4.0],
            [4.1, 4.0],
            [4.0, 4.1],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    metrics = evaluate_clusters(X, labels, silhouette_sample_size=None)

    assert metrics["n_clusters"] == 2.0
    assert -1.0 <= metrics["silhouette"] <= 1.0
    assert metrics["davies_bouldin"] >= 0.0


def test_metrics_table_filters_non_numeric_and_excluded_keys() -> None:
    table = metrics_table(
        {
            "test": {
                "accuracy": 0.91234,
                "loss": np.float64(0.12345),
                "classification_report": "text report",
                "confusion_matrix": np.array([[1, 0], [0, 1]]),
                "debug": 99,
            }
        },
        exclude_keys=["debug"],
    )

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["accuracy", "loss"]
    assert table.loc["test", "accuracy"] == 0.9123
