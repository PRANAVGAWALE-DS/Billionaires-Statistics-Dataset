"""
tests/test_loader.py
--------------------
Unit tests for billionaires.data.loader:
    load_raw, validate_schema, clean
"""

import numpy as np
import pandas as pd
import pytest

from billionaires.data.loader import clean, load_raw, validate_schema

# ── load_raw ──────────────────────────────────────────────────────────────


class TestLoadRaw:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_raw(tmp_path / "ghost.csv")

    def test_returns_dataframe(self, tmp_path, raw_df):
        p = tmp_path / "data.csv"
        raw_df.to_csv(p, index=False)
        df = load_raw(p)
        assert isinstance(df, pd.DataFrame)

    def test_shape_preserved(self, tmp_path, raw_df):
        p = tmp_path / "data.csv"
        raw_df.to_csv(p, index=False)
        df = load_raw(p)
        assert df.shape == raw_df.shape

    def test_accepts_path_object(self, tmp_path, raw_df):
        from pathlib import Path

        p = tmp_path / "data.csv"
        raw_df.to_csv(p, index=False)
        df = load_raw(Path(p))
        assert len(df) == len(raw_df)


# ── validate_schema ───────────────────────────────────────────────────────


class TestValidateSchema:
    def test_passes_on_valid_df(self, raw_df):
        validate_schema(raw_df)  # must not raise

    def test_raises_on_missing_required_col(self, raw_df):
        df = raw_df.drop(columns=["selfMade"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_schema(df)

    def test_raises_on_multiple_missing_cols(self, raw_df):
        df = raw_df.drop(columns=["selfMade", "finalWorth"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_schema(df)

    def test_extra_columns_allowed(self, raw_df):
        df = raw_df.copy()
        df["extra_col"] = 0
        validate_schema(df)  # must not raise


# ── clean ─────────────────────────────────────────────────────────────────


class TestClean:
    def test_returns_dataframe(self, clean_df):
        assert isinstance(clean_df, pd.DataFrame)

    def test_no_null_finalworth(self, raw_df):
        df = raw_df.copy()
        df.loc[0, "finalWorth"] = np.nan
        result = clean(df)
        assert result["finalWorth"].isna().sum() == 0

    def test_null_finalworth_rows_dropped(self, raw_df):
        df = raw_df.copy()
        df.loc[:2, "finalWorth"] = np.nan  # 3 null rows
        result = clean(df)
        assert len(result) == len(df) - 3

    def test_selfmade_is_integer(self, clean_df):
        assert pd.api.types.is_integer_dtype(clean_df["selfMade"])

    def test_selfmade_only_zero_or_one(self, clean_df):
        assert set(clean_df["selfMade"].unique()).issubset({0, 1})

    def test_age_no_nulls_after_imputation(self, raw_df):
        df = raw_df.copy()
        df.loc[:5, "age"] = np.nan
        result = clean(df)
        assert result["age"].isna().sum() == 0

    def test_duplicates_removed(self, raw_df):
        df = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
        result = clean(df)
        assert len(result) <= len(df) - 10

    def test_returns_independent_copy(self, raw_df):
        original_len = len(raw_df)
        clean(raw_df)
        assert len(raw_df) == original_len  # original not mutated

    def test_numeric_coercion_finalworth(self, raw_df):
        df = raw_df.copy()
        df["finalWorth"] = df["finalWorth"].astype(str)
        result = clean(df)
        assert pd.api.types.is_float_dtype(result["finalWorth"])
