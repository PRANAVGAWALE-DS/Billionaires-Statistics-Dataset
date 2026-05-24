"""
api/predictor.py
----------------
BillionairesPredictor — loads all serialised artifacts from
``data/processed/`` and provides three inference methods.

Unit convention
---------------
All public methods accept ``finalWorth`` in **billion USD** (the natural
unit for users and the API surface).  ``_build_row`` converts to
**million USD** before calling ``build_features``, because the model was
trained on the raw CSV which stores ``finalWorth`` in millions.

  User input  →  API / Streamlit (billions)
  Conversion  →  _build_row multiplies by 1 000
  Model input →  build_features(finalWorth_millions)  [matches training]

Inference data flow
-------------------
Caller provides:  finalWorth (B), age, category, country, gender
                        │
                  _build_row()
                  │  1. finalWorth_M = finalWorth_B * 1000   ← unit fix
                  │  2. build_features(encode=False)
                  │     derives: log_worth, wealth_per_decade,
                  │              age_group, continent
                  │  3. FeatureEncoder.transform()
                  │     adds: category_enc, country_enc,
                  │           gender_enc, continent_enc
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
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from billionaires.features.engineer import FeatureEncoder

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
            finalWorth=5.2, age=52,             # finalWorth in billion USD
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
        # Feature column lists — populated from feature_cols.json in load()
        self._clf_cols: list[str] = []
        self._reg_cols: list[str] = []
        self._cluster_cols: list[str] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> BillionairesPredictor:
        """Load all artifact files.  Call once at application startup.

        FIX H3 — ``feature_cols.json`` is the authoritative source of truth
        for which columns each model uses.  Column lists are read from the
        JSON manifest rather than from the installed package's module
        functions.  This ensures the inference layer always uses exactly the
        feature set that was frozen at training time, regardless of any
        subsequent changes to ``engineer.py``.

        Raises
        ------
        FileNotFoundError
            If any required artifact is missing from ``processed_dir``.
            Run ``python pipeline.py --cluster-k 4`` to generate them.
        """
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

        # FIX H3 — read column lists from the serialised manifest, not from
        # the live package.  The manifest was written by pipeline.py at
        # training time; it is the canonical contract between training and
        # serving.
        self._feature_cols = json.loads((d / "feature_cols.json").read_text())
        self._clf_cols = self._feature_cols["clf"]
        self._reg_cols = self._feature_cols["reg"]
        self._cluster_cols = self._feature_cols["cluster"]

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
        1. Convert ``finalWorth`` from billion USD → million USD.
           The raw CSV (and therefore the training pipeline) stores
           ``finalWorth`` in **millions**; the API surface uses **billions**.
           Passing billions directly would push all features out of the
           training distribution (log_worth ≈ 0–5 instead of 7–12).
        2. Construct a one-row DataFrame with the raw input fields.
        3. Call :func:`build_features` (encode=False) — adds log_worth,
           wealth_per_decade, age_group, continent.
        4. Call :meth:`FeatureEncoder.transform` — adds *_enc columns.

        Parameters
        ----------
        finalWorth : float
            Net worth in **billion USD** (public API unit).
        selfMade : int
            0 or 1.  Pass the classifier's prediction for regressor /
            clusterer calls; the value is a feature, not the target.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with all engineered + encoded columns.
        """
        from billionaires.features.engineer import build_features

        # FIX C1 — convert billions → millions to match the training scale.
        # The CSV stores finalWorth in millions; log1p(5.2B) ≈ 1.84 is
        # completely outside the training range, while log1p(5200M) ≈ 8.56
        # is well within it.  birthYear is also removed (M3) — it is not a
        # model feature and was previously dead computation.
        finalWorth_millions = float(finalWorth) * 1_000

        raw = pd.DataFrame(
            [
                {
                    "personName": "__api__",
                    "finalWorth": finalWorth_millions,
                    "age": float(age),
                    "category": str(category),
                    "country": str(country),
                    "gender": str(gender),
                    "selfMade": int(selfMade),
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

        Parameters
        ----------
        finalWorth : float
            Net worth in billion USD.

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

        Parameters
        ----------
        finalWorth : float
            Net worth in billion USD (used as a feature, not the target).

        Returns
        -------
        dict
            log_worth_predicted, worth_billion_usd, selfMade_used

        Notes
        -----
        The regressor has R²=0.048 on the test set — it explains ~5% of
        log-worth variance.  Predictions reflect population-level trends
        (industry, geography, age) and should be treated as indicative
        directional estimates rather than precise point forecasts.
        """
        self._check_loaded()
        df = self._build_row(finalWorth, age, category, country, gender, selfMade)
        X = df[self._reg_cols].to_numpy()

        log_pred = float(self._reg.predict(X)[0])
        # The model predicts log1p(finalWorth_millions).
        # expm1 converts back to millions; divide by 1 000 for billions.
        worth_pred_billions = float(np.expm1(log_pred)) / 1_000
        return {
            "log_worth_predicted": round(log_pred, 4),
            "worth_billion_usd": round(worth_pred_billions, 4),
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

        Parameters
        ----------
        finalWorth : float
            Net worth in billion USD.

        Returns
        -------
        dict
            cluster (int), n_clusters (int), silhouette (float)

        Notes
        -----
        The clusterer was fitted on the full dataset (train + val + test)
        for exploratory segmentation purposes.  This is intentional and
        documented in ``pipeline.py:_run_clusterer``.
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

        Parameters
        ----------
        finalWorth : float
            Net worth in billion USD.

        Returns
        -------
        dict
            self_made, worth, cluster — each a nested result dict.

        Notes
        -----
        Prefer this endpoint over calling ``/predict/worth`` and
        ``/predict/cluster`` individually — the combined path runs the
        classifier exactly once and reuses ``selfMade`` for all downstream
        models.  Calling the individual endpoints back-to-back runs the
        classifier twice unnecessarily (see L2).
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
