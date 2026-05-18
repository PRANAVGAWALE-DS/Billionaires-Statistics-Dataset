"""
api/predictor.py
----------------
BillionairesPredictor — loads all serialised artifacts from
``data/processed/`` and provides three inference methods.

Inference data flow
-------------------
Caller provides:  finalWorth, age, category, country, gender
                        │
                  build_features(encode=False)
                        │   derives: log_worth, wealth_per_decade,
                        │            age_group, continent
                        │
                  FeatureEncoder.transform()
                        │   adds: category_enc, country_enc,
                        │         gender_enc, continent_enc
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    Classifier      Regressor    Clusterer
    → P(selfMade)  → log_worth  → cluster_id
                   uses selfMade  uses selfMade
                   from classifier from classifier

selfMade is never provided by the caller — it is always inferred
from the classifier so predictions are fully self-contained.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROCESSED_DIR = Path("data/processed")

_REQUIRED_ARTIFACTS = [
    "classifier.joblib",
    "regressor.joblib",
    "clusterer.joblib",
    "feature_encoder.joblib",
    "feature_cols.json",
    "metrics.json",
]


class BillionairesPredictor:
    """Loads serialised artifacts and exposes predict_* methods.

    Usage
    -----
    ::

        predictor = BillionairesPredictor()
        predictor.load()                        # call once at startup

        result = predictor.predict_all(
            finalWorth=5.2, age=52,
            category="Technology", country="United States", gender="M",
        )

    Parameters
    ----------
    processed_dir : Path
        Directory containing the serialised artifacts produced by
        ``python pipeline.py``.  Defaults to ``data/processed``.
    """

    def __init__(self, processed_dir: Path = _PROCESSED_DIR) -> None:
        self.processed_dir = Path(processed_dir)
        self._clf = None
        self._reg = None
        self._clusterer = None
        self._enc: FeatureEncoder | None = None
        self._feature_cols: dict[str, list[str]] | None = None
        self._metrics: dict | None = None
        # Cached feature column lists (set in load())
        self._clf_cols: list[str] = []
        self._reg_cols: list[str] = []
        self._cluster_cols: list[str] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> "BillionairesPredictor":
        """Load all artifact files.  Call once at application startup.

        Raises
        ------
        FileNotFoundError
            If any required artifact is missing from ``processed_dir``.
            Run ``python pipeline.py --cluster-k 4`` to generate them.
        """
        # Lazy import — keeps BillionairesPredictor definable even if
        # the billionaires package has a transient import error.
        from billionaires.features.engineer import (
            FeatureEncoder,
            get_clf_features,
            get_cluster_features,
            get_reg_features,
        )

        self._clf_cols = get_clf_features()
        self._reg_cols = get_reg_features()
        self._cluster_cols = get_cluster_features()

        d = self.processed_dir
        missing = [f for f in _REQUIRED_ARTIFACTS if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifact(s) in '{d}': {missing}\n"
                "Run `python pipeline.py --cluster-k 4` to generate them."
            )

        self._clf = joblib.load(d / "classifier.joblib")
        self._reg = joblib.load(d / "regressor.joblib")
        self._clusterer = joblib.load(d / "clusterer.joblib")
        self._enc = joblib.load(d / "feature_encoder.joblib")
        self._feature_cols = json.loads((d / "feature_cols.json").read_text())
        self._metrics = json.loads((d / "metrics.json").read_text())

        logger.info(
            "BillionairesPredictor loaded — clf=%s  reg=%s  clusterer k=%d",
            type(self._clf).__name__,
            type(self._reg).__name__,
            self._clusterer.k_,
        )
        return self

    @property
    def loaded(self) -> bool:
        """True once :meth:`load` has completed successfully."""
        return all(
            x is not None for x in [self._clf, self._reg, self._clusterer, self._enc]
        )

    @property
    def artifact_names(self) -> list[str]:
        """Names of artifact files that were loaded."""
        return _REQUIRED_ARTIFACTS if self.loaded else []

    @property
    def metrics(self) -> dict:
        """Stored evaluation metrics from the last pipeline run."""
        return self._metrics or {}

    # ── Feature construction ──────────────────────────────────────────────

    def _build_row(
        self,
        finalWorth: float,
        age: float,
        category: str,
        country: str,
        gender: str,
        selfMade: int,
    ) -> pd.DataFrame:
        """Reproduce pipeline feature engineering for a single input row.

        Steps
        -----
        1. Construct a one-row DataFrame with the raw input fields.
        2. Call :func:`build_features` (encode=False) — adds log_worth,
           wealth_per_decade, age_group, continent.
        3. Call :meth:`FeatureEncoder.transform` — adds *_enc columns.

        Parameters
        ----------
        selfMade : int
            0 or 1.  Pass the classifier's prediction for regressor /
            clusterer calls; the value is a feature, not the target.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with all engineered + encoded columns.
        """
        from billionaires.features.engineer import build_features

        raw = pd.DataFrame(
            [
                {
                    "personName": "__api__",
                    "finalWorth": float(finalWorth),
                    "age": float(age),
                    "category": str(category),
                    "country": str(country),
                    "gender": str(gender),
                    "selfMade": int(selfMade),
                    "birthYear": float(2024 - age),
                }
            ]
        )
        df_feat = build_features(raw, encode=False)
        df_enc = self._enc.transform(df_feat)
        return df_enc

    # ── Prediction methods ────────────────────────────────────────────────

    def predict_self_made(
        self,
        finalWorth: float,
        age: float,
        category: str,
        country: str,
        gender: str,
    ) -> dict:
        """Predict self-made probability from the XGBoost classifier.

        Returns
        -------
        dict
            probability (float), prediction (0|1), label (str)
        """
        self._check_loaded()
        df = self._build_row(finalWorth, age, category, country, gender, selfMade=0)
        X = df[self._clf_cols].to_numpy()

        prob = float(self._clf.predict_proba(X)[0, 1])
        pred = int(self._clf.predict(X)[0])
        return {
            "probability": round(prob, 4),
            "prediction": pred,
            "label": "Self-Made" if pred == 1 else "Inherited",
        }

    def predict_worth(
        self,
        finalWorth: float,
        age: float,
        category: str,
        country: str,
        gender: str,
        selfMade: int,
    ) -> dict:
        """Predict net worth from the XGBoost regressor.

        Returns
        -------
        dict
            log_worth_predicted, worth_billion_usd, selfMade_used
        """
        self._check_loaded()
        df = self._build_row(finalWorth, age, category, country, gender, selfMade)
        X = df[self._reg_cols].to_numpy()

        log_pred = float(self._reg.predict(X)[0])
        worth_pred = float(np.expm1(log_pred))
        return {
            "log_worth_predicted": round(log_pred, 4),
            "worth_billion_usd": round(worth_pred, 4),
            "selfMade_used": selfMade,
        }

    def predict_cluster(
        self,
        finalWorth: float,
        age: float,
        category: str,
        country: str,
        gender: str,
        selfMade: int,
    ) -> dict:
        """Assign the input to a wealth segment cluster.

        Returns
        -------
        dict
            cluster (int), n_clusters (int), silhouette (float)
        """
        self._check_loaded()
        df = self._build_row(finalWorth, age, category, country, gender, selfMade)
        X = df[self._cluster_cols].to_numpy()

        cluster = int(self._clusterer.predict(X)[0])
        silhouette = float(
            self._metrics.get("clusterer", {}).get("silhouette", float("nan"))
        )
        return {
            "cluster": cluster,
            "n_clusters": int(self._clusterer.k_),
            "silhouette": round(silhouette, 4),
        }

    def predict_all(
        self,
        finalWorth: float,
        age: float,
        category: str,
        country: str,
        gender: str,
    ) -> dict:
        """Run all three models in sequence and return combined output.

        selfMade is inferred from the classifier and propagated as a
        feature to the regressor and clusterer.

        Returns
        -------
        dict
            self_made, worth, cluster — each a nested result dict.
        """
        self._check_loaded()
        sm = self.predict_self_made(finalWorth, age, category, country, gender)
        sm_pred = sm["prediction"]  # classifier output → regressor/cluster feature

        worth = self.predict_worth(finalWorth, age, category, country, gender, sm_pred)
        cluster = self.predict_cluster(
            finalWorth, age, category, country, gender, sm_pred
        )

        return {"self_made": sm, "worth": worth, "cluster": cluster}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_loaded(self) -> None:
        if not self.loaded:
            raise RuntimeError(
                "BillionairesPredictor is not loaded. Call .load() first."
            )
