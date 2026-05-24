"""
billionaires.features.engineer
-------------------------------
All feature engineering lives here.  Notebooks and model modules call
:func:`build_features` rather than redefining transformations inline.

Production ML pipelines should use :class:`FeatureEncoder` for split-aware
categorical encoding.  :func:`build_features` retains the legacy
:class:`~sklearn.preprocessing.LabelEncoder` path *only* for EDA notebooks
that never split the data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)

# Country → continent lookup (expanded to cover dataset's full country list)
_CONTINENT_MAP: dict[str, str] = {
    # North America
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    # South America
    "Brazil": "South America",
    "Colombia": "South America",
    "Chile": "South America",
    "Argentina": "South America",
    "Venezuela": "South America",
    "Peru": "South America",
    # Europe
    "United Kingdom": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Italy": "Europe",
    "Switzerland": "Europe",
    "Sweden": "Europe",
    "Spain": "Europe",
    "Netherlands": "Europe",
    "Russia": "Europe",
    "Norway": "Europe",
    "Denmark": "Europe",
    "Austria": "Europe",
    "Belgium": "Europe",
    "Finland": "Europe",
    "Ireland": "Europe",
    "Portugal": "Europe",
    "Poland": "Europe",
    "Czech Republic": "Europe",
    "Greece": "Europe",
    "Hungary": "Europe",
    "Romania": "Europe",
    "Ukraine": "Europe",
    "Turkey": "Europe",
    # Asia
    "China": "Asia",
    "India": "Asia",
    "Japan": "Asia",
    "Hong Kong": "Asia",
    "Singapore": "Asia",
    "South Korea": "Asia",
    "Taiwan": "Asia",
    "Thailand": "Asia",
    "Indonesia": "Asia",
    "Israel": "Asia",
    "Saudi Arabia": "Asia",
    "United Arab Emirates": "Asia",
    "Malaysia": "Asia",
    "Philippines": "Asia",
    "Vietnam": "Asia",
    "Pakistan": "Asia",
    "Bangladesh": "Asia",
    "Kazakhstan": "Asia",
    "Uzbekistan": "Asia",
    "Azerbaijan": "Asia",
    "Kuwait": "Asia",
    "Qatar": "Asia",
    "Oman": "Asia",
    "Bahrain": "Asia",
    "Lebanon": "Asia",
    "Jordan": "Asia",
    "Sri Lanka": "Asia",
    "Myanmar": "Asia",
    # Africa
    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Egypt": "Africa",
    "Kenya": "Africa",
    "Morocco": "Africa",
    "Tanzania": "Africa",
    "Algeria": "Africa",
    # Oceania
    "Australia": "Oceania",
    "New Zealand": "Oceania",
}

_AGE_BINS = [0, 40, 55, 65, 80, 120]
_AGE_LABELS = ["<40", "40-54", "55-64", "65-79", "80+"]

# Categorical columns consumed by both models
_ENCODE_COLS = ["category", "country", "gender", "continent"]


# ── Base feature engineering (no encoding) ────────────────────────────────


def build_features(df: pd.DataFrame, *, encode: bool = True) -> pd.DataFrame:
    """Add engineered columns to a **copy** of *df*.

    New columns
    -----------
    log_worth         : log1p(finalWorth) — normalises heavy right skew
    age_group         : ordinal age bin (<40 … 80+)
    wealth_per_decade : finalWorth / (age / 10)  — productivity proxy
    continent         : mapped from country (expanded to ~80 countries)
    {col}_enc         : label-encoded categoricals (only when encode=True)

    Parameters
    ----------
    df     : Cleaned DataFrame from :func:`billionaires.data.loader.clean`.
    encode : If True (default), append ``{col}_enc`` columns via
             :class:`~sklearn.preprocessing.LabelEncoder` fitted on *df*.
             Set ``encode=False`` and use :class:`FeatureEncoder` instead
             for any ML pipeline that performs a train/test split — doing
             so prevents preprocessing leakage into the held-out set.

    Returns
    -------
    pd.DataFrame
        Extended copy with all new columns appended.
    """
    df = df.copy()

    # log-transform the target — removes right skew for visualisation & regression
    df["log_worth"] = np.log1p(df["finalWorth"])

    # ordinal age bucketing
    df["age_group"] = pd.cut(df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, right=False)

    # wealth per decade of life (avoid div/0 for age==0, though unlikely)
    decades = (df["age"] / 10).replace(0, np.nan)
    df["wealth_per_decade"] = df["finalWorth"] / decades

    # continent (expanded map — falls back to "Other" for any gap)
    df["continent"] = df["country"].map(_CONTINENT_MAP).fillna("Other")

    new_base = ["log_worth", "age_group", "wealth_per_decade", "continent"]

    if encode:
        # ── Legacy LabelEncoder path (EDA notebooks only) ─────────────────
        # Fitted on the full df — safe for visualisation, NOT for ML splits.
        # For split-aware encoding use FeatureEncoder.fit(X_train).transform(X).
        enc_cols: list[str] = []
        for col in _ENCODE_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                enc_cols.append(f"{col}_enc")
        logger.info(
            "build_features complete — added %d columns (%s + %s)",
            len(new_base) + len(enc_cols),
            new_base,
            enc_cols,
        )
    else:
        logger.info(
            "build_features complete — added %d base columns (encoding skipped)",
            len(new_base),
        )

    return df


# ── Split-aware encoder for production ML pipelines ───────────────────────


class FeatureEncoder:
    """Fit-once / transform-many categorical encoder for ML pipelines.

    Wraps :class:`~sklearn.preprocessing.OrdinalEncoder` with
    ``handle_unknown="use_encoded_value", unknown_value=-1`` so unseen
    categories at inference time don't raise an error.

    Usage
    -----
    ::

        df_feat = build_features(df_clean, encode=False)

        X_train_raw = df_feat.loc[train_idx, get_clf_features_raw()]
        X_val_raw   = df_feat.loc[val_idx,   get_clf_features_raw()]
        X_test_raw  = df_feat.loc[test_idx,  get_clf_features_raw()]

        enc = FeatureEncoder()
        X_train = enc.fit_transform(X_train_raw)
        X_val   = enc.transform(X_val_raw)
        X_test  = enc.transform(X_test_raw)

    Parameters
    ----------
    cols : list[str] | None
        Columns to encode.  Defaults to ``_ENCODE_COLS`` if None.
    """

    def __init__(self, cols: list[str] | None = None) -> None:
        self.cols: list[str] = cols if cols is not None else list(_ENCODE_COLS)
        self._encoder: OrdinalEncoder | None = None
        self._present_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """Fit the encoder on *df* (training split only).

        Parameters
        ----------
        df : DataFrame containing the categorical columns in self.cols.
        """
        self._present_cols = [c for c in self.cols if c in df.columns]
        self._encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )
        self._encoder.fit(df[self._present_cols].astype(str))
        logger.info(
            "FeatureEncoder fitted on %d rows — cols: %s",
            len(df),
            self._present_cols,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with ``{col}_enc`` columns appended.

        Parameters
        ----------
        df : DataFrame with the same categorical columns used in fit().
        """
        self._check_fitted()
        df = df.copy()
        encoded = self._encoder.transform(df[self._present_cols].astype(str))
        for i, col in enumerate(self._present_cols):
            df[f"{col}_enc"] = encoded[:, i]
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        return self.fit(df).transform(df)

    def _check_fitted(self) -> None:
        if self._encoder is None:
            raise RuntimeError("Call .fit() before .transform().")


# ── Feature list accessors ────────────────────────────────────────────────


def get_clf_features() -> list[str]:
    """Ordered feature list for :class:`~billionaires.models.classifier.SelfMadeClassifier`.

    Excludes the raw target (``selfMade``) and ``finalWorth`` (represented
    via ``log_worth``).  ``wealth_per_decade`` is a legitimate predictor
    for the classifier — net worth is known before the self-made question
    is asked and does not constitute leakage.

    Note on ``log_worth``
    ---------------------
    ``log_worth = log1p(finalWorth)`` is a monotone transformation of the
    input feature ``finalWorth``, not of the classifier's binary target
    ``selfMade``.  Using it in the classifier is valid: we are predicting
    *how* someone became wealthy, given *how much* they are worth.
    This differs from the regressor case, where ``log_worth`` IS the
    target — making it illegal as a regressor feature.
    """
    base = ["age", "log_worth", "wealth_per_decade"]
    encoded = [f"{c}_enc" for c in _ENCODE_COLS]
    return base + encoded


def get_reg_features() -> list[str]:
    """Ordered feature list for :class:`~billionaires.models.regressor.WorthRegressor`.

    Excludes ``finalWorth``, ``log_worth`` (the prediction target), and
    ``wealth_per_decade`` (``finalWorth / (age/10)`` — direct target
    leakage that would inflate R²).
    """
    base = ["age", "selfMade"]
    encoded = [f"{c}_enc" for c in _ENCODE_COLS]
    return base + encoded


def get_cluster_features() -> list[str]:
    """Feature list for :class:`~billionaires.models.clusterer.BillionaireClusterer`.

    Uses log-scale wealth to avoid scale dominance from raw ``finalWorth``.
    Excludes ``wealth_per_decade`` (too correlated with ``log_worth`` +
    ``age`` to add independent cluster structure).
    """
    base = ["age", "log_worth", "selfMade"]
    encoded = [f"{c}_enc" for c in _ENCODE_COLS]
    return base + encoded
