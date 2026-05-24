"""
billionaires.models.evaluate
-----------------------------
Centralised, testable evaluation utilities for all three model types.

Keeps metric computation entirely out of notebooks — notebooks call these
functions and display results rather than defining metric logic inline.

Functions
---------
evaluate_classifier  : full suite for binary classification
evaluate_regressor   : MAE / RMSE / R² on log scale + optional dollar-scale MAPE
evaluate_clusters    : silhouette + Davies-Bouldin (no ground truth needed)
metrics_table        : format a nested results dict into a display DataFrame

Usage
-----
::

    from billionaires.models.evaluate import (
        evaluate_classifier,
        evaluate_regressor,
        evaluate_clusters,
        metrics_table,
    )

    clf_metrics = evaluate_classifier(y_test, y_pred, y_prob, split="test")
    reg_metrics = evaluate_regressor(y_test, y_pred_log, exponentiated=True, split="test")
    cluster_metrics = evaluate_clusters(X_scaled, labels)

    print(metrics_table({"clf": clf_metrics, "reg": reg_metrics}))
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


# ── Classification ────────────────────────────────────────────────────────


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    label_names: list[str] | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Full binary-classification metrics dictionary.

    Parameters
    ----------
    y_true      : Ground-truth labels (0/1).
    y_pred      : Hard predictions.
    y_prob      : Positive-class probabilities (column 1 of predict_proba).
    label_names : Class names for the classification report.
                  Defaults to ``["Inherited", "Self-Made"]``.
    split       : Label used in log messages (e.g. "val", "test").

    Returns
    -------
    dict
        Keys: roc_auc, accuracy, precision, recall, f1,
              classification_report (str), confusion_matrix (ndarray).
    """
    if label_names is None:
        label_names = ["Inherited", "Self-Made"]

    metrics: dict[str, Any] = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "classification_report": classification_report(y_true, y_pred, target_names=label_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    logger.info(
        "[%s] CLF — AUC: %.4f  Acc: %.4f  Precision: %.4f  Recall: %.4f  F1: %.4f",
        split,
        metrics["roc_auc"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics


# ── Regression ────────────────────────────────────────────────────────────


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    exponentiated: bool = False,
    split: str = "test",
) -> dict[str, float]:
    """Regression metrics on the log scale, with optional dollar-scale extras.

    Parameters
    ----------
    y_true        : True ``log1p(finalWorth)`` values.
    y_pred        : Predicted ``log1p(finalWorth)`` values.
    exponentiated : If True, also compute MAE, RMSE, and MAPE on the
                    original dollar (billion USD) scale via ``np.expm1``.
    split         : Label used in log messages.

    Returns
    -------
    dict
        Always contains: mae, rmse, r2.
        When exponentiated=True, also contains: mae_raw, rmse_raw, mape_raw.

        ``mape_raw`` is clamped by adding 1e-8 to the denominator to avoid
        division by zero for any edge case where true worth rounds to zero.
    """
    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

    if exponentiated:
        y_true_raw = np.expm1(y_true)
        y_pred_raw = np.expm1(y_pred)
        mape = float(np.mean(np.abs((y_true_raw - y_pred_raw) / (np.abs(y_true_raw) + 1e-8))) * 100)
        metrics["mae_raw"] = float(mean_absolute_error(y_true_raw, y_pred_raw))
        metrics["rmse_raw"] = float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
        metrics["mape_raw"] = mape

    logger.info(
        "[%s] REG — R²: %.4f  MAE: %.4f  RMSE: %.4f%s",
        split,
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
        f"  MAPE(raw): {metrics['mape_raw']:.2f}%" if exponentiated else "",
    )
    return metrics


# ── Clustering ────────────────────────────────────────────────────────────


def evaluate_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    *,
    split: str = "full",
    silhouette_sample_size: int | None = 500,
) -> dict[str, float]:
    """Unsupervised cluster quality metrics (no ground truth required).

    Parameters
    ----------
    X_scaled : np.ndarray, shape (n_samples, n_features)
        **Already-scaled** feature matrix (pass the StandardScaler output,
        not raw features — silhouette is distance-based).
    labels : np.ndarray, shape (n_samples,)
        Cluster assignments from :class:`~billionaires.models.clusterer.BillionaireClusterer`.
    split : str
        Label used in log messages.

    Returns
    -------
    dict
        Keys: silhouette, davies_bouldin, n_clusters.

        Higher silhouette (max 1.0) is better.
        Lower Davies-Bouldin (min 0.0) is better.
    """
    n_clusters = int(len(np.unique(labels)))
    if n_clusters < 2:
        logger.warning(
            "[%s] CLUSTER — only %d cluster found; silhouette undefined.",
            split,
            n_clusters,
        )
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "n_clusters": n_clusters,
        }

    metrics: dict[str, float] = {
        "silhouette": float(
            silhouette_score(
                X_scaled,
                labels,
                sample_size=silhouette_sample_size,
                random_state=42,
            )
        ),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "n_clusters": float(n_clusters),
    }
    logger.info(
        "[%s] CLUSTER — k=%d  silhouette: %.4f  davies_bouldin: %.4f",
        split,
        n_clusters,
        metrics["silhouette"],
        metrics["davies_bouldin"],
    )
    return metrics


# ── Formatting ────────────────────────────────────────────────────────────


def metrics_table(
    results: dict[str, dict[str, float]],
    *,
    exclude_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Format a nested ``{split: {metric: value}}`` dict into a display DataFrame.

    Non-numeric values (e.g. ``classification_report``, ``confusion_matrix``)
    are excluded automatically unless explicitly listed in ``exclude_keys``.

    Parameters
    ----------
    results      : Nested dict as returned by the evaluate_* functions.
    exclude_keys : Additional keys to drop from the table.

    Returns
    -------
    pd.DataFrame
        Rows = splits, columns = metrics, values rounded to 4 decimal places.

    Example
    -------
    ::

        table = metrics_table({
            "val":  evaluate_classifier(y_val,  p_val,  prob_val,  split="val"),
            "test": evaluate_classifier(y_test, p_test, prob_test, split="test"),
        })
        display(table)
    """
    skip: set[str] = {"classification_report", "confusion_matrix"}
    if exclude_keys:
        skip.update(exclude_keys)

    tidy: dict[str, dict[str, float]] = {}
    for split_name, split_metrics in results.items():
        tidy[split_name] = {
            k: v
            for k, v in split_metrics.items()
            if k not in skip and isinstance(v, (int, float, np.floating, np.integer))
        }

    return pd.DataFrame(tidy).T.round(4)
