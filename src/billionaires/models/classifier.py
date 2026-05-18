"""
billionaires.models.classifier
-------------------------------
XGBoost binary classifier that predicts whether a billionaire is
self-made (1) or inherited (0).  Hyperparameters are tuned via Optuna.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# XGBoost 2.x requires early_stopping_rounds to live in .fit(), not in the
# constructor — setting it there with eval_set=None raises ValueError.
_EARLY_STOPPING_ROUNDS = 20


@dataclass
class SelfMadeClassifier:
    """XGBoost classifier for self-made vs inherited wealth prediction.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials for HPO (default 40).
    cv_folds : int
        K-fold cross-validation splits during HPO (default 5).
    seed : int
        Random seed for reproducibility (default 42).

    Attributes
    ----------
    best_params_ : dict
        Best hyperparameters found by Optuna.
    model_ : xgb.XGBClassifier
        Fitted model (available after calling :meth:`fit`).
    study_ : optuna.Study
        Full Optuna study object (available after :meth:`tune`).
    """

    n_trials: int = 40
    cv_folds: int = 5
    seed: int = 42
    device: str = "cpu"  # set "cuda" to use GPU (RTX 3050 / any CUDA GPU)

    best_params_: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    model_: xgb.XGBClassifier | None = field(default=None, init=False, repr=False)
    study_: optuna.Study | None = field(default=None, init=False, repr=False)

    # ── HPO ───────────────────────────────────────────────────────────────

    def tune(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Run Optuna HPO on (X, y) using stratified K-fold CV.

        Optimises ROC-AUC.  Results stored in :attr:`best_params_`
        and :attr:`study_`.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)
        y : np.ndarray  shape (n_samples,)  — binary 0/1

        Returns
        -------
        dict
            Best hyperparameter dictionary.
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)

        def _objective(trial: optuna.Trial) -> float:
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                gamma=trial.suggest_float("gamma", 0.0, 0.5),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                random_state=self.seed,
                device=self.device,
            )
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(_objective, n_trials=self.n_trials, show_progress_bar=True)

        self.study_ = study
        self.best_params_ = study.best_params
        logger.info("HPO complete — best ROC-AUC (CV): %.4f", study.best_value)
        return self.best_params_

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "SelfMadeClassifier":
        """Fit the classifier with :attr:`best_params_`.

        If ``best_params_`` is empty (i.e. :meth:`tune` was not called),
        XGBoost defaults are used.

        ``early_stopping_rounds`` is passed to :meth:`xgb.XGBClassifier.fit`
        only when a validation set is provided — XGBoost 2.x raises
        ``ValueError`` if it is set without a matching ``eval_set``.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val     : optional validation set for early stopping

        Returns
        -------
        self
        """
        # early_stopping_rounds must NOT go in the constructor in XGBoost 2.x
        # when eval_set may be absent — keep it in fit() kwargs only.
        constructor_params = {
            **self.best_params_,
            "random_state": self.seed,
            "device": self.device,
            "eval_metric": "logloss",
        }
        # XGBoost 2.x: early_stopping_rounds belongs in the constructor,
        # NOT in fit().  Only set it when an eval_set will be supplied.
        if X_val is not None and y_val is not None:
            constructor_params["early_stopping_rounds"] = _EARLY_STOPPING_ROUNDS
        self.model_ = xgb.XGBClassifier(**constructor_params)

        fit_kwargs: dict[str, Any] = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model_.fit(X_train, y_train, **fit_kwargs)
        logger.info("Model fitted on %d samples", len(y_train))
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict_proba(X)

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Return a metrics dictionary for the given split.

        Keys
        ----
        roc_auc, accuracy, classification_report (str),
        confusion_matrix (ndarray)

        .. tip::
            For richer metrics (precision, recall, F1) use
            :func:`billionaires.models.evaluate.evaluate_classifier`.
        """
        self._check_fitted()
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        return {
            "roc_auc": roc_auc_score(y, y_prob),
            "classification_report": classification_report(
                y, y_pred, target_names=["Inherited", "Self-Made"]
            ),
            "confusion_matrix": confusion_matrix(y, y_pred),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("Call .fit() before predicting.")
