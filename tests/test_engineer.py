"""
tests/test_engineer.py
----------------------
Unit tests for billionaires.features.engineer:
    build_features, FeatureEncoder,
    get_clf_features, get_reg_features, get_cluster_features
"""

import numpy as np
import pandas as pd
import pytest

from billionaires.features.engineer import (
    FeatureEncoder,
    build_features,
    get_clf_features,
    get_cluster_features,
    get_reg_features,
)

# ── build_features ────────────────────────────────────────────────────────


class TestBuildFeatures:
    def test_returns_dataframe(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=False)
        assert isinstance(df, pd.DataFrame)

    def test_original_not_mutated(self, sample_df_clean):
        cols_before = set(sample_df_clean.columns)
        build_features(sample_df_clean, encode=False)
        assert set(sample_df_clean.columns) == cols_before

    def test_adds_log_worth(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=False)
        assert "log_worth" in df.columns

    def test_log_worth_finite(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=False)
        assert np.isfinite(df["log_worth"]).all()

    def test_adds_age_group(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=False)
        assert "age_group" in df.columns

    def test_adds_wealth_per_decade(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=False)
        assert "wealth_per_decade" in df.columns

    def test_adds_continent(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=False)
        assert "continent" in df.columns

    def test_continent_no_nulls(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=False)
        # unknown countries fall back to "Other" — no NaN
        assert df["continent"].isna().sum() == 0

    def test_encode_true_adds_enc_cols(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=True)
        for col in ["category_enc", "country_enc", "gender_enc", "continent_enc"]:
            assert col in df.columns, f"missing {col}"

    def test_encode_false_no_enc_cols(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=False)
        for col in ["category_enc", "country_enc", "gender_enc", "continent_enc"]:
            assert col not in df.columns, f"unexpected {col}"

    def test_row_count_preserved(self, sample_df_clean):
        df = build_features(sample_df_clean, encode=False)
        assert len(df) == len(sample_df_clean)


# ── FeatureEncoder ────────────────────────────────────────────────────────


class TestFeatureEncoder:
    @pytest.fixture
    def feat_df(self, sample_df_clean):
        return build_features(sample_df_clean, encode=False)

    def test_fit_transform_adds_enc_cols(self, feat_df):
        enc = FeatureEncoder()
        df_enc = enc.fit_transform(feat_df)
        for col in ["category_enc", "country_enc", "gender_enc", "continent_enc"]:
            assert col in df_enc.columns

    def test_transform_matches_fit_transform(self, feat_df):
        enc = FeatureEncoder()
        df_ft = enc.fit_transform(feat_df)
        enc2 = FeatureEncoder()
        enc2.fit(feat_df)
        df_t = enc2.transform(feat_df)
        pd.testing.assert_frame_equal(
            df_ft[["category_enc"]].reset_index(drop=True),
            df_t[["category_enc"]].reset_index(drop=True),
        )

    def test_unknown_category_returns_minus_one(self, feat_df):
        enc = FeatureEncoder()
        enc.fit(feat_df)
        df_unseen = feat_df.copy()
        df_unseen["category"] = "Completely__Unknown__XYZ"
        df_enc = enc.transform(df_unseen)
        assert (df_enc["category_enc"] == -1).all()

    def test_enc_values_are_numeric(self, feat_df):
        enc = FeatureEncoder()
        df_enc = enc.fit_transform(feat_df)
        for col in ["category_enc", "country_enc", "gender_enc", "continent_enc"]:
            assert pd.api.types.is_float_dtype(
                df_enc[col]
            ) or pd.api.types.is_integer_dtype(df_enc[col])

    def test_original_not_mutated(self, feat_df):
        cols_before = set(feat_df.columns)
        enc = FeatureEncoder()
        enc.fit_transform(feat_df)
        assert set(feat_df.columns) == cols_before

    def test_not_fitted_raises(self, feat_df):
        enc = FeatureEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(feat_df)

    def test_row_count_preserved(self, feat_df):
        enc = FeatureEncoder()
        df_enc = enc.fit_transform(feat_df)
        assert len(df_enc) == len(feat_df)


# ── Feature list accessors ────────────────────────────────────────────────


class TestFeatureLists:
    def test_clf_features_is_list_of_strings(self):
        feats = get_clf_features()
        assert isinstance(feats, list)
        assert all(isinstance(f, str) for f in feats)

    def test_clf_features_nonempty(self):
        assert len(get_clf_features()) > 0

    def test_clf_features_excludes_target(self):
        # selfMade is the classification target — must not be a feature
        assert "selfMade" not in get_clf_features()

    def test_clf_features_includes_log_worth(self):
        assert "log_worth" in get_clf_features()

    def test_reg_features_is_list_of_strings(self):
        feats = get_reg_features()
        assert isinstance(feats, list)
        assert all(isinstance(f, str) for f in feats)

    def test_reg_features_excludes_log_worth_target(self):
        # log_worth is the regression target — must not be a feature
        assert "log_worth" not in get_reg_features()

    def test_reg_features_excludes_wealth_per_decade_leakage(self):
        # wealth_per_decade = finalWorth / (age/10) — direct target leakage
        assert "wealth_per_decade" not in get_reg_features()

    def test_cluster_features_is_list_of_strings(self):
        feats = get_cluster_features()
        assert isinstance(feats, list)
        assert all(isinstance(f, str) for f in feats)

    def test_no_overlap_reg_leakage(self):
        # Ensure reg and clf feature lists don't accidentally share the target
        clf = set(get_clf_features())
        reg = set(get_reg_features())
        assert "selfMade" not in clf
        assert "log_worth" not in reg
