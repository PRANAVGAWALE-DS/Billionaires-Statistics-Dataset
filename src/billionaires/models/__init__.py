"""
billionaires.models
--------------------
Public API for all model classes and evaluation utilities.

Usage
-----
from billionaires.models import (
    SelfMadeClassifier,
    WorthRegressor,
    BillionaireClusterer,
    evaluate_classifier,
    evaluate_regressor,
    evaluate_clusters,
    metrics_table,
)
"""

from billionaires.models.classifier import SelfMadeClassifier
from billionaires.models.clusterer import BillionaireClusterer
from billionaires.models.evaluate import (
    evaluate_classifier,
    evaluate_clusters,
    evaluate_regressor,
    metrics_table,
)
from billionaires.models.regressor import WorthRegressor

__all__ = [
    "SelfMadeClassifier",
    "WorthRegressor",
    "BillionaireClusterer",
    "evaluate_classifier",
    "evaluate_regressor",
    "evaluate_clusters",
    "metrics_table",
]
