"""
billionaires.features.engineer
-------------------------------
All feature engineering lives here.  Notebooks and model modules call
:func:`build_features` rather than redefining transformations inline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Country → continent lookup (top countries covered)
_CONTINENT_MAP: dict[str, str] = {
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Brazil": "South America",
    "Colombia": "South America",
    "Chile": "South America",
    "Argentina": "South America",
    "United Kingdom": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Italy": "Europe",
    "Switzerland": "Europe",
    "Sweden": "Europe",
    "Spain": "Europe",
    "Netherlands": "Europe",
    "Russia": "Europe",
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
    "Australia": "Oceania",
    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Egypt": "Africa",
}

_AGE_BINS = [0, 40, 55, 65, 80, 120]
_AGE_LABELS = ["<40", "40–54", "55–64", "65–79", "80+"]

# Categorical columns to label-encode for ML
_ENCODE_COLS = ["category", "country", "gender", "continent"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to a **copy** of *df*.

    New columns
    -----------
    log_worth         : log1p(finalWorth) — normalises heavy right skew
    age_group         : ordinal age bin (<40 … 80+)
    wealth_per_decade : finalWorth / (age / 10)  — productivity proxy
    continent         : mapped from country
    {col}_enc         : label-encoded version of each categorical in
                        ``category``, ``country``, ``gender``, ``continent``

    .. note::
        Label encoders are fitted on the full DataFrame passed in.
        For training pipelines, ensure ``build_features`` is called
        before the train/test split, and treat ``{col}_enc`` columns
        as fixed for this dataset.  Encoding unseen categories at
        inference time will raise a ``ValueError``; a production
        pipeline should replace :class:`~sklearn.preprocessing.LabelEncoder`
        with :class:`~sklearn.preprocessing.OrdinalEncoder` fitted on
        the training split only, using ``handle_unknown="use_encoded_value"``.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from :func:`billionaires.data.loader.clean`.

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

    # continent
    df["continent"] = df["country"].map(_CONTINENT_MAP).fillna("Other")

    # label-encode categoricals for ML consumption
    for col in _ENCODE_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    logger.info(
        "Feature engineering complete — added %d new columns",
        len(["log_worth", "age_group", "wealth_per_decade", "continent"])
        + sum(1 for c in _ENCODE_COLS if c in df.columns),
    )
    return df


def get_clf_features() -> list[str]:
    """Return the ordered feature list used by :class:`SelfMadeClassifier`.

    Does **not** include the target (``selfMade``).  ``finalWorth`` is
    excluded in its raw form since ``log_worth`` already captures the
    same information on a normalised scale and avoids sensitivity to
    extreme outliers.  Net worth is a legitimate predictor of wealth
    origin and does not constitute leakage — it is known before the
    classification question is asked.
    """
    base = ["age", "log_worth", "wealth_per_decade"]
    encoded = [f"{c}_enc" for c in _ENCODE_COLS]
    return base + encoded


def get_reg_features() -> list[str]:
    """Return the feature list used by :class:`WorthRegressor`.

    Excludes ``finalWorth``, ``log_worth`` (the target), and
    ``wealth_per_decade`` (which is ``finalWorth / (age/10)`` and
    therefore directly encodes the target — including it would cause
    leakage and inflate R²).
    """
    base = ["age", "selfMade"]
    encoded = [f"{c}_enc" for c in _ENCODE_COLS]
    return base + encoded
