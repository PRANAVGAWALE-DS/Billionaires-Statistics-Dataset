"""
tests/test_clusterer.py
-----------------------
Unit tests for BillionaireClusterer.

Covers
------
- fit() happy path with explicit k and auto elbow search
- transform_pca() output shape and dtype
- predict() on new data — cluster IDs in range
- cluster_profiles() DataFrame structure
- diagnostics() contents
- _check_fitted guard on every public method
- n_init=10 is set (H4 regression guard)
- elbow search selects best k by silhouette
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from billionaires.features.engineer import get_cluster_features
from billionaires.models.clusterer import BillionaireClusterer

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cluster_X(sample_df_encoded: pd.DataFrame) -> np.ndarray:
    feat_cols = get_cluster_features()
    return sample_df_encoded[feat_cols].dropna().to_numpy()


@pytest.fixture(scope="module")
def clusterer_fixed_k(cluster_X: np.ndarray) -> BillionaireClusterer:
    c = BillionaireClusterer(seed=42)
    c.fit(cluster_X, k=3)
    return c


@pytest.fixture(scope="module")
def clusterer_auto_k(cluster_X: np.ndarray) -> BillionaireClusterer:
    """Auto k-selection via elbow search over k_range=(2,4)."""
    c = BillionaireClusterer(k_range=(2, 4), seed=42)
    c.fit(cluster_X)
    return c


# ── fit() ─────────────────────────────────────────────────────────────────


class TestFit:
    def test_k_set_correctly(self, clusterer_fixed_k: BillionaireClusterer) -> None:
        assert clusterer_fixed_k.k_ == 3

    def test_model_not_none(self, clusterer_fixed_k: BillionaireClusterer) -> None:
        assert clusterer_fixed_k.model_ is not None

    def test_scaler_fitted(self, clusterer_fixed_k: BillionaireClusterer) -> None:
        assert clusterer_fixed_k.scaler_ is not None

    def test_pca_fitted(self, clusterer_fixed_k: BillionaireClusterer) -> None:
        assert clusterer_fixed_k.pca_ is not None

    def test_labels_shape(
        self, clusterer_fixed_k: BillionaireClusterer, cluster_X: np.ndarray
    ) -> None:
        assert clusterer_fixed_k.labels_.shape == (len(cluster_X),)

    def test_labels_in_range(self, clusterer_fixed_k: BillionaireClusterer) -> None:
        assert set(np.unique(clusterer_fixed_k.labels_)).issubset(set(range(3)))

    def test_returns_self(self, cluster_X: np.ndarray) -> None:
        c = BillionaireClusterer(seed=42)
        result = c.fit(cluster_X, k=2)
        assert result is c

    def test_n_init_is_10(self, clusterer_fixed_k: BillionaireClusterer) -> None:
        """Regression guard for H4 — n_init must be 10, not 1."""
        assert clusterer_fixed_k.model_.n_init == 10


# ── Elbow search (auto k) ─────────────────────────────────────────────────


class TestElbowSearch:
    def test_k_in_range(self, clusterer_auto_k: BillionaireClusterer) -> None:
        assert 2 <= clusterer_auto_k.k_ <= 4

    def test_inertias_populated(self, clusterer_auto_k: BillionaireClusterer) -> None:
        assert len(clusterer_auto_k.inertias_) == 3  # k_range (2, 4)
        for k in range(2, 5):
            assert k in clusterer_auto_k.inertias_
            assert clusterer_auto_k.inertias_[k] > 0

    def test_silhouettes_populated(
        self, clusterer_auto_k: BillionaireClusterer
    ) -> None:
        assert len(clusterer_auto_k.silhouettes_) == 3
        for k in range(2, 5):
            assert -1.0 <= clusterer_auto_k.silhouettes_[k] <= 1.0

    def test_best_k_maximises_silhouette(
        self, clusterer_auto_k: BillionaireClusterer
    ) -> None:
        best_from_dict = max(
            clusterer_auto_k.silhouettes_,
            key=lambda k: clusterer_auto_k.silhouettes_[k],
        )
        assert clusterer_auto_k.k_ == best_from_dict

    def test_diagnostics_returns_dataframe(
        self, clusterer_auto_k: BillionaireClusterer
    ) -> None:
        df = clusterer_auto_k.diagnostics()
        assert isinstance(df, pd.DataFrame)
        assert "inertia" in df.columns
        assert "silhouette" in df.columns

    def test_diagnostics_empty_when_k_fixed(
        self, clusterer_fixed_k: BillionaireClusterer
    ) -> None:
        df = clusterer_fixed_k.diagnostics()
        assert df.empty


# ── transform_pca() ───────────────────────────────────────────────────────


class TestTransformPCA:
    def test_output_shape(
        self, clusterer_fixed_k: BillionaireClusterer, cluster_X: np.ndarray
    ) -> None:
        xy = clusterer_fixed_k.transform_pca(cluster_X)
        assert xy.shape == (len(cluster_X), 2)

    def test_output_dtype_float(
        self, clusterer_fixed_k: BillionaireClusterer, cluster_X: np.ndarray
    ) -> None:
        xy = clusterer_fixed_k.transform_pca(cluster_X)
        assert np.issubdtype(xy.dtype, np.floating)

    def test_single_row(
        self, clusterer_fixed_k: BillionaireClusterer, cluster_X: np.ndarray
    ) -> None:
        xy = clusterer_fixed_k.transform_pca(cluster_X[:1])
        assert xy.shape == (1, 2)


# ── predict() ─────────────────────────────────────────────────────────────


class TestPredict:
    def test_output_shape(
        self, clusterer_fixed_k: BillionaireClusterer, cluster_X: np.ndarray
    ) -> None:
        labels = clusterer_fixed_k.predict(cluster_X)
        assert labels.shape == (len(cluster_X),)

    def test_labels_in_range(
        self, clusterer_fixed_k: BillionaireClusterer, cluster_X: np.ndarray
    ) -> None:
        labels = clusterer_fixed_k.predict(cluster_X)
        assert set(np.unique(labels)).issubset(set(range(3)))

    def test_single_row(
        self, clusterer_fixed_k: BillionaireClusterer, cluster_X: np.ndarray
    ) -> None:
        label = clusterer_fixed_k.predict(cluster_X[:1])
        assert label.shape == (1,)
        assert 0 <= int(label[0]) < 3


# ── cluster_profiles() ────────────────────────────────────────────────────


class TestClusterProfiles:
    def test_returns_dataframe(
        self,
        clusterer_fixed_k: BillionaireClusterer,
        sample_df_encoded: pd.DataFrame,
    ) -> None:
        feat_cols = get_cluster_features()
        df_sub = sample_df_encoded[feat_cols].dropna().reset_index(drop=True)
        profiles = clusterer_fixed_k.cluster_profiles(df_sub, feat_cols)
        assert isinstance(profiles, pd.DataFrame)

    def test_index_contains_all_cluster_ids(
        self,
        clusterer_fixed_k: BillionaireClusterer,
        sample_df_encoded: pd.DataFrame,
    ) -> None:
        feat_cols = get_cluster_features()
        df_sub = sample_df_encoded[feat_cols].dropna().reset_index(drop=True)
        profiles = clusterer_fixed_k.cluster_profiles(df_sub, feat_cols)
        assert set(profiles.index) == set(range(3))

    def test_has_mean_and_median_stats(
        self,
        clusterer_fixed_k: BillionaireClusterer,
        sample_df_encoded: pd.DataFrame,
    ) -> None:
        feat_cols = get_cluster_features()
        df_sub = sample_df_encoded[feat_cols].dropna().reset_index(drop=True)
        profiles = clusterer_fixed_k.cluster_profiles(df_sub, feat_cols)
        col_level_1 = profiles.columns.get_level_values(1).unique().tolist()
        assert "mean" in col_level_1
        assert "median" in col_level_1


# ── _check_fitted guards ──────────────────────────────────────────────────


class TestNotFittedGuard:
    def test_transform_pca_raises(self, cluster_X: np.ndarray) -> None:
        c = BillionaireClusterer()
        with pytest.raises(RuntimeError, match="not fitted"):
            c.transform_pca(cluster_X)

    def test_predict_raises(self, cluster_X: np.ndarray) -> None:
        c = BillionaireClusterer()
        with pytest.raises(RuntimeError, match="not fitted"):
            c.predict(cluster_X)

    def test_cluster_profiles_raises(self, sample_df_encoded: pd.DataFrame) -> None:
        c = BillionaireClusterer()
        feat_cols = get_cluster_features()
        df_sub = sample_df_encoded[feat_cols].dropna().reset_index(drop=True)
        with pytest.raises(RuntimeError, match="not fitted"):
            c.cluster_profiles(df_sub, feat_cols)
