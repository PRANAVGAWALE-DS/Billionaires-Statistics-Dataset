"""
tests/test_regressor.py
-----------------------
Unit tests for billionaires.models.regressor.WorthRegressor.

All tests use fit() without tune() to avoid Optuna overhead.
The model runs on a 120-row synthetic DataFrame — fast and deterministic.
"""

import numpy as np
import pytest

from billionaires.features.engineer import (
    FeatureEncoder,
    build_features,
    get_reg_features,
)
from billionaires.models.regressor import WorthRegressor

# ── Shared fixture ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def reg_arrays(clean_df):
    """(X, y) arrays ready for WorthRegressor.fit()."""
    df_feat = build_features(clean_df, encode=False)
    enc = FeatureEncoder()
    df_enc = enc.fit_transform(df_feat)
    feats = [f for f in get_reg_features() if f in df_enc.columns]
    X = df_enc[feats].to_numpy()
    y = df_enc["log_worth"].to_numpy()
    return X, y


@pytest.fixture(scope="module")
def fitted_reg(reg_arrays):
    """WorthRegressor fitted on reg_arrays (no HPO)."""
    X, y = reg_arrays
    reg = WorthRegressor(n_trials=2, cv_folds=2, seed=42)
    reg.fit(X, y)
    return reg


# ── Instantiation ──────────────────────────────────────────────────────────


class TestWorthRegressorInit:
    def test_default_instantiation(self):
        reg = WorthRegressor()
        assert reg.n_trials == 40
        assert reg.cv_folds == 5
        assert reg.seed == 42

    def test_custom_params(self):
        reg = WorthRegressor(n_trials=5, cv_folds=3, seed=1)
        assert reg.n_trials == 5
        assert reg.cv_folds == 3
        assert reg.seed == 1

    def test_model_none_before_fit(self):
        reg = WorthRegressor()
        assert reg.model_ is None


# ── fit() ─────────────────────────────────────────────────────────────────


class TestWorthRegressorFit:
    def test_fit_returns_self(self, reg_arrays):
        X, y = reg_arrays
        reg = WorthRegressor(seed=42)
        result = reg.fit(X, y)
        assert result is reg

    def test_model_not_none_after_fit(self, fitted_reg):
        assert fitted_reg.model_ is not None

    def test_fit_with_val_set(self, reg_arrays):
        X, y = reg_arrays
        mid = len(X) // 2
        reg = WorthRegressor(seed=42)
        reg.fit(X[:mid], y[:mid], X_val=X[mid:], y_val=y[mid:])
        assert reg.model_ is not None

    def test_fit_without_val_set(self, reg_arrays):
        """fit() with no val set must not raise (early_stopping_rounds guarded)."""
        X, y = reg_arrays
        reg = WorthRegressor(seed=42)
        reg.fit(X, y)  # X_val=None, y_val=None — must not raise
        assert reg.model_ is not None


# ── predict() ─────────────────────────────────────────────────────────────


class TestWorthRegressorPredict:
    def test_predict_shape(self, fitted_reg, reg_arrays):
        X, y = reg_arrays
        preds = fitted_reg.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_log_scale_finite(self, fitted_reg, reg_arrays):
        X, _ = reg_arrays
        preds = fitted_reg.predict(X)
        assert np.isfinite(preds).all()

    def test_predict_exponentiate_shape(self, fitted_reg, reg_arrays):
        X, y = reg_arrays
        preds = fitted_reg.predict(X, exponentiate=True)
        assert preds.shape == (len(y),)

    def test_predict_exponentiate_nonnegative(self, fitted_reg, reg_arrays):
        """expm1(predictions) should be >= -1; for real billionaire data > 0."""
        X, _ = reg_arrays
        preds = fitted_reg.predict(X, exponentiate=True)
        assert (preds >= -1.0).all()

    def test_predict_exponentiate_differs_from_log(self, fitted_reg, reg_arrays):
        X, _ = reg_arrays
        log_preds = fitted_reg.predict(X, exponentiate=False)
        raw_preds = fitted_reg.predict(X, exponentiate=True)
        assert not np.allclose(log_preds, raw_preds)

    def test_predict_before_fit_raises(self):
        reg = WorthRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict(np.zeros((5, 3)))


# ── evaluate() ────────────────────────────────────────────────────────────


class TestWorthRegressorEvaluate:
    def test_evaluate_returns_dict(self, fitted_reg, reg_arrays):
        X, y = reg_arrays
        metrics = fitted_reg.evaluate(X, y)
        assert isinstance(metrics, dict)

    def test_evaluate_has_mae(self, fitted_reg, reg_arrays):
        X, y = reg_arrays
        assert "mae" in fitted_reg.evaluate(X, y)

    def test_evaluate_has_rmse(self, fitted_reg, reg_arrays):
        X, y = reg_arrays
        assert "rmse" in fitted_reg.evaluate(X, y)

    def test_evaluate_has_r2(self, fitted_reg, reg_arrays):
        X, y = reg_arrays
        assert "r2" in fitted_reg.evaluate(X, y)

    def test_mae_nonnegative(self, fitted_reg, reg_arrays):
        X, y = reg_arrays
        metrics = fitted_reg.evaluate(X, y)
        assert metrics["mae"] >= 0.0

    def test_rmse_nonnegative(self, fitted_reg, reg_arrays):
        X, y = reg_arrays
        metrics = fitted_reg.evaluate(X, y)
        assert metrics["rmse"] >= 0.0

    def test_rmse_geq_mae(self, fitted_reg, reg_arrays):
        """RMSE >= MAE is a mathematical identity (Cauchy-Schwarz)."""
        X, y = reg_arrays
        metrics = fitted_reg.evaluate(X, y)
        assert metrics["rmse"] >= metrics["mae"] - 1e-9
