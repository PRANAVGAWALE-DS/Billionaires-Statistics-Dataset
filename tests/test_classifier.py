"""
tests/test_classifier.py
------------------------
Unit tests for billionaires.models.classifier.SelfMadeClassifier.

All tests use fit() without tune() to avoid Optuna overhead.
The model runs on a 120-row synthetic DataFrame — fast and deterministic.
"""

import numpy as np
import pytest

from billionaires.features.engineer import (
    FeatureEncoder,
    build_features,
    get_clf_features,
)
from billionaires.models.classifier import SelfMadeClassifier

# ── Shared fixture ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def clf_arrays(sample_df_clean):
    """(X, y) arrays ready for SelfMadeClassifier.fit()."""
    df_feat = build_features(sample_df_clean, encode=False)
    enc = FeatureEncoder()
    df_enc = enc.fit_transform(df_feat)
    feats = [f for f in get_clf_features() if f in df_enc.columns]
    X = df_enc[feats].to_numpy()
    y = df_enc["selfMade"].to_numpy()
    return X, y


@pytest.fixture(scope="module")
def fitted_clf(clf_arrays):
    """SelfMadeClassifier fitted on clf_arrays (no HPO)."""
    X, y = clf_arrays
    clf = SelfMadeClassifier(n_trials=2, cv_folds=2, seed=42)
    clf.fit(X, y)
    return clf


# ── Instantiation ──────────────────────────────────────────────────────────


class TestSelfMadeClassifierInit:
    def test_default_instantiation(self):
        clf = SelfMadeClassifier()
        assert clf.n_trials == 40
        assert clf.cv_folds == 5
        assert clf.seed == 42

    def test_custom_params(self):
        clf = SelfMadeClassifier(n_trials=10, cv_folds=3, seed=0)
        assert clf.n_trials == 10
        assert clf.cv_folds == 3
        assert clf.seed == 0

    def test_model_none_before_fit(self):
        clf = SelfMadeClassifier()
        assert clf.model_ is None


# ── fit() ─────────────────────────────────────────────────────────────────


class TestSelfMadeClassifierFit:
    def test_fit_returns_self(self, clf_arrays):
        X, y = clf_arrays
        clf = SelfMadeClassifier(seed=42)
        result = clf.fit(X, y)
        assert result is clf

    def test_model_not_none_after_fit(self, fitted_clf):
        assert fitted_clf.model_ is not None

    def test_fit_with_val_set(self, clf_arrays):
        X, y = clf_arrays
        mid = len(X) // 2
        clf = SelfMadeClassifier(seed=42)
        clf.fit(X[:mid], y[:mid], X_val=X[mid:], y_val=y[mid:])
        assert clf.model_ is not None

    def test_fit_without_val_set(self, clf_arrays):
        """fit() with no val set must not raise (early_stopping_rounds guarded)."""
        X, y = clf_arrays
        clf = SelfMadeClassifier(seed=42)
        clf.fit(X, y)  # X_val=None, y_val=None — must not raise
        assert clf.model_ is not None


# ── predict() ─────────────────────────────────────────────────────────────


class TestSelfMadeClassifierPredict:
    def test_predict_shape(self, fitted_clf, clf_arrays):
        X, y = clf_arrays
        preds = fitted_clf.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_only_binary(self, fitted_clf, clf_arrays):
        X, _ = clf_arrays
        preds = fitted_clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, fitted_clf, clf_arrays):
        X, y = clf_arrays
        proba = fitted_clf.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_predict_proba_sums_to_one(self, fitted_clf, clf_arrays):
        X, _ = clf_arrays
        proba = fitted_clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, fitted_clf, clf_arrays):
        X, _ = clf_arrays
        proba = fitted_clf.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_before_fit_raises(self):
        clf = SelfMadeClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.zeros((5, 3)))

    def test_predict_proba_before_fit_raises(self):
        clf = SelfMadeClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_proba(np.zeros((5, 3)))


# ── evaluate() ────────────────────────────────────────────────────────────


class TestSelfMadeClassifierEvaluate:
    def test_evaluate_returns_dict(self, fitted_clf, clf_arrays):
        X, y = clf_arrays
        metrics = fitted_clf.evaluate(X, y)
        assert isinstance(metrics, dict)

    def test_evaluate_has_roc_auc(self, fitted_clf, clf_arrays):
        X, y = clf_arrays
        metrics = fitted_clf.evaluate(X, y)
        assert "roc_auc" in metrics

    def test_evaluate_has_confusion_matrix(self, fitted_clf, clf_arrays):
        X, y = clf_arrays
        metrics = fitted_clf.evaluate(X, y)
        assert "confusion_matrix" in metrics

    def test_evaluate_has_classification_report(self, fitted_clf, clf_arrays):
        X, y = clf_arrays
        metrics = fitted_clf.evaluate(X, y)
        assert "classification_report" in metrics

    def test_roc_auc_in_valid_range(self, fitted_clf, clf_arrays):
        X, y = clf_arrays
        metrics = fitted_clf.evaluate(X, y)
        assert 0.0 <= metrics["roc_auc"] <= 1.0
