"""
billionaires.models.regressor
------------------------------
XGBoost regressor that predicts ``log(1 + finalWorth)``.
Predictions are optionally back-transformed to the original dollar scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# XGBoost 2.x: early_stopping_rounds belongs in the constructor, NOT in
# fit().  Only set it when an eval_set will be supplied; omitting it when
# no eval_set is present avoids the ValueError XGBoost 2.x raises.
_EARLY_STOPPING_ROUNDS = 20


@dataclass
class WorthRegressor:
    """XGBoost regressor for log(1 + finalWorth) prediction.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials (default 40).
    cv_folds : int
        K-fold CV splits during HPO (default 5).
    seed : int
        Random seed (default 42).

    Attributes
    ----------
    best_params_ : dict
    model_       : xgb.XGBRegressor  (after fit)
    study_       : optuna.Study       (after tune)
    """

    n_trials: int = 40
    cv_folds: int = 5
    seed: int = 42
    device: str = "cpu"  # set "cuda" to use GPU (RTX 3050 / any CUDA GPU)

    best_params_: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    model_: xgb.XGBRegressor | None = field(default=None, init=False, repr=False)
    study_: optuna.Study | None = field(default=None, init=False, repr=False)

    # ── HPO ───────────────────────────────────────────────────────────────

    def tune(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Optuna HPO on (X, y).  Optimises R² via K-fold CV."""
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)

        def _objective(trial: optuna.Trial) -> float:
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 600),
                max_depth=trial.suggest_int("max_depth", 3, 9),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                random_state=self.seed,
                device=self.device,
            )
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=1)
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(_objective, n_trials=self.n_trials, show_progress_bar=True)

        self.study_ = study
        self.best_params_ = study.best_params
        logger.info("HPO complete — best R² (CV): %.4f", study.best_value)
        return self.best_params_

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> WorthRegressor:
        """Fit the regressor with :attr:`best_params_`.

        ``early_stopping_rounds`` is passed to the constructor only when a
        validation set is provided — XGBoost 2.x raises ``ValueError`` if it
        is set without a matching ``eval_set``.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val     : optional validation set for early stopping

        Returns
        -------
        self
        """
        constructor_params = {
            **self.best_params_,
            "random_state": self.seed,
            "device": self.device,
            "eval_metric": "rmse",
        }
        # XGBoost 2.x: early_stopping_rounds belongs in the constructor,
        # NOT in fit().  Only set it when an eval_set will be supplied.
        if X_val is not None and y_val is not None:
            constructor_params["early_stopping_rounds"] = _EARLY_STOPPING_ROUNDS
        self.model_ = xgb.XGBRegressor(**constructor_params)

        fit_kwargs: dict[str, Any] = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model_.fit(X_train, y_train, **fit_kwargs)
        logger.info("Regressor fitted on %d samples", len(y_train))
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray, *, exponentiate: bool = False) -> np.ndarray:
        """Predict log(1 + finalWorth).

        Parameters
        ----------
        X           : feature matrix
        exponentiate: if True, apply ``np.expm1`` to convert back to
                      the original billion-dollar scale.
        """
        self._check_fitted()
        y_hat = self.model_.predict(X)
        return np.expm1(y_hat) if exponentiate else y_hat

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Return MAE, RMSE, and R² on the log scale.

        .. tip::
            For dollar-scale MAPE and richer reporting use
            :func:`billionaires.models.evaluate.evaluate_regressor`.
        """
        self._check_fitted()
        y_pred = self.predict(X)
        return {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": r2_score(y, y_pred),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("Call .fit() before predicting.")
