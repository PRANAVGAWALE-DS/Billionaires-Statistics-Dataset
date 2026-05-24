"""
tests/conftest.py
-----------------
Shared pytest fixtures used across all test modules.

Fixtures
--------
sample_df_raw       : minimal raw DataFrame with required columns
sample_df_clean     : cleaned version via loader.clean()
sample_df_feat      : feature-engineered (encode=False)
sample_df_encoded   : feature-engineered + FeatureEncoder applied
fitted_encoder      : FeatureEncoder fitted on sample_df_feat
fitted_classifier   : SelfMadeClassifier fitted on sample data (no HPO)
fitted_regressor    : WorthRegressor fitted on sample data (no HPO)
fitted_clusterer    : BillionaireClusterer fitted on sample data (k=2)
mock_processed_dir  : tmp_path directory containing all serialised artifacts
loaded_predictor    : BillionairesPredictor with all artifacts loaded
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from billionaires.data.loader import clean
from billionaires.features.engineer import (
    FeatureEncoder,
    build_features,
    get_clf_features,
    get_cluster_features,
    get_reg_features,
)
from billionaires.models.classifier import SelfMadeClassifier
from billionaires.models.clusterer import BillionaireClusterer
from billionaires.models.regressor import WorthRegressor

# ── Deterministic seed ────────────────────────────────────────────────────
_SEED = 42
np.random.seed(_SEED)

# ── Raw data factory ──────────────────────────────────────────────────────

_N = 120  # enough rows for stratified 80/20 split in classifier/regressor


def _make_raw_df(n: int = _N, seed: int = _SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = ["Technology", "Finance", "Manufacturing", "Healthcare", "Energy"]
    countries = ["United States", "China", "India", "Germany", "United Kingdom"]
    genders = ["M", "F"]
    return pd.DataFrame(
        {
            "personName": [f"Person_{i}" for i in range(n)],
            # finalWorth in millions (raw CSV scale)
            "finalWorth": rng.uniform(1_000, 100_000, n),
            "category": rng.choice(categories, n),
            "country": rng.choice(countries, n),
            "age": rng.uniform(30, 90, n),
            "gender": rng.choice(genders, n),
            "selfMade": rng.integers(0, 2, n),
            "birthYear": (2024 - rng.uniform(30, 90, n)).astype(int),
        }
    )


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sample_df_raw() -> pd.DataFrame:
    return _make_raw_df()


@pytest.fixture(scope="session")
def sample_df_clean(sample_df_raw: pd.DataFrame) -> pd.DataFrame:
    return clean(sample_df_raw)


@pytest.fixture(scope="session")
def sample_df_feat(sample_df_clean: pd.DataFrame) -> pd.DataFrame:
    return build_features(sample_df_clean, encode=False)


@pytest.fixture(scope="session")
def fitted_encoder(sample_df_feat: pd.DataFrame) -> FeatureEncoder:
    enc = FeatureEncoder()
    enc.fit(sample_df_feat)
    return enc


@pytest.fixture(scope="session")
def sample_df_encoded(
    sample_df_feat: pd.DataFrame,
    fitted_encoder: FeatureEncoder,
) -> pd.DataFrame:
    return fitted_encoder.transform(sample_df_feat)


@pytest.fixture(scope="session")
def fitted_classifier(sample_df_encoded: pd.DataFrame) -> SelfMadeClassifier:
    """SelfMadeClassifier fitted without HPO (XGBoost defaults, fast)."""
    feat_cols = get_clf_features()
    X = sample_df_encoded[feat_cols].to_numpy()
    y = sample_df_encoded["selfMade"].to_numpy()
    clf = SelfMadeClassifier(n_trials=5, seed=_SEED)
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="session")
def fitted_regressor(sample_df_encoded: pd.DataFrame) -> WorthRegressor:
    """WorthRegressor fitted without HPO (XGBoost defaults, fast)."""
    feat_cols = get_reg_features()
    X = sample_df_encoded[feat_cols].to_numpy()
    y = sample_df_encoded["log_worth"].to_numpy()
    reg = WorthRegressor(n_trials=5, seed=_SEED)
    reg.fit(X, y)
    return reg


@pytest.fixture(scope="session")
def fitted_clusterer(sample_df_encoded: pd.DataFrame) -> BillionaireClusterer:
    """BillionaireClusterer fitted with fixed k=2 (fast, no elbow search)."""
    feat_cols = get_cluster_features()
    X = sample_df_encoded[feat_cols].dropna().to_numpy()
    clusterer = BillionaireClusterer(seed=_SEED)
    clusterer.fit(X, k=2)
    return clusterer


@pytest.fixture(scope="session")
def mock_processed_dir(
    tmp_path_factory: pytest.TempPathFactory,
    fitted_classifier: SelfMadeClassifier,
    fitted_regressor: WorthRegressor,
    fitted_clusterer: BillionaireClusterer,
    fitted_encoder: FeatureEncoder,
) -> Path:
    """Write all required artifacts to a temporary directory.

    Returns a Path that BillionairesPredictor.load() can consume.
    """
    d = tmp_path_factory.mktemp("processed")

    joblib.dump(fitted_classifier, d / "classifier.joblib")
    joblib.dump(fitted_regressor, d / "regressor.joblib")
    joblib.dump(fitted_clusterer, d / "clusterer.joblib")
    joblib.dump(fitted_encoder, d / "feature_encoder.joblib")

    feature_cols = {
        "clf": get_clf_features(),
        "reg": get_reg_features(),
        "cluster": get_cluster_features(),
    }
    (d / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))

    metrics = {
        "classifier": {"test": {"roc_auc": 0.75, "f1": 0.72}},
        "regressor": {"test": {"r2": 0.05, "mae": 0.6}},
        "clusterer": {"silhouette": 0.20, "davies_bouldin": 1.5, "n_clusters": 2.0},
    }
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return d


@pytest.fixture(scope="session")
def loaded_predictor(mock_processed_dir: Path):
    """BillionairesPredictor with all artifacts loaded from mock_processed_dir."""
    from api.predictor import BillionairesPredictor

    p = BillionairesPredictor(processed_dir=mock_processed_dir)
    p.load()
    return p
