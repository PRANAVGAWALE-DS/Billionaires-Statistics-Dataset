"""
tests/test_predictor.py
-----------------------
Unit and integration tests for BillionairesPredictor.

Covers
------
- Artifact loading (happy path + missing artifacts)
- _build_row unit conversion (C1 regression guard)
- Feature column alignment with feature_cols.json (H3)
- predict_self_made output contract
- predict_worth output contract + unit sanity
- predict_cluster output contract
- predict_all combined output
- Unfitted guard (_check_loaded)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.predictor import BillionairesPredictor

# ── Load / lifecycle ───────────────────────────────────────────────────────


class TestPredicatorLoad:
    def test_load_succeeds(self, loaded_predictor: BillionairesPredictor) -> None:
        assert loaded_predictor.loaded is True

    def test_artifact_names_populated(self, loaded_predictor: BillionairesPredictor) -> None:
        names = loaded_predictor.artifact_names
        assert len(names) > 0
        assert "classifier.joblib" in names
        assert "feature_cols.json" in names

    def test_metrics_populated(self, loaded_predictor: BillionairesPredictor) -> None:
        m = loaded_predictor.metrics
        assert "classifier" in m
        assert "regressor" in m
        assert "clusterer" in m

    def test_load_raises_on_missing_artifacts(self, tmp_path: Path) -> None:
        """FileNotFoundError must be raised when artifacts are absent."""
        p = BillionairesPredictor(processed_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="Missing artifact"):
            p.load()

    def test_not_loaded_before_load(self, mock_processed_dir: Path) -> None:
        """loaded property must be False before .load() is called."""
        p = BillionairesPredictor(processed_dir=mock_processed_dir)
        assert p.loaded is False

    def test_check_loaded_raises_before_load(self, mock_processed_dir: Path) -> None:
        p = BillionairesPredictor(processed_dir=mock_processed_dir)
        with pytest.raises(RuntimeError, match="not loaded"):
            p.predict_self_made(
                finalWorth=5.0,
                age=50,
                category="Technology",
                country="United States",
                gender="M",
            )


# ── Feature column alignment (H3) ─────────────────────────────────────────


class TestFeatureColumnAlignment:
    def test_clf_cols_match_json(
        self, loaded_predictor: BillionairesPredictor, mock_processed_dir: Path
    ) -> None:
        """_clf_cols must match what was written to feature_cols.json at training time."""
        on_disk = json.loads((mock_processed_dir / "feature_cols.json").read_text())
        assert loaded_predictor._clf_cols == on_disk["clf"]

    def test_reg_cols_match_json(
        self, loaded_predictor: BillionairesPredictor, mock_processed_dir: Path
    ) -> None:
        on_disk = json.loads((mock_processed_dir / "feature_cols.json").read_text())
        assert loaded_predictor._reg_cols == on_disk["reg"]

    def test_cluster_cols_match_json(
        self, loaded_predictor: BillionairesPredictor, mock_processed_dir: Path
    ) -> None:
        on_disk = json.loads((mock_processed_dir / "feature_cols.json").read_text())
        assert loaded_predictor._cluster_cols == on_disk["cluster"]


# ── _build_row unit conversion — C1 regression guard ──────────────────────


class TestBuildRowUnitConversion:
    """Regression tests that catch the billion/million unit mismatch (C1).

    The training CSV stores finalWorth in millions.  The API accepts billions.
    _build_row must multiply by 1 000 before calling build_features so that
    log_worth sits in the training distribution [~6.9, ~12.4].

    If the conversion is removed, log_worth will be ~[0.7, 5.3] — completely
    outside the training range — and these assertions will fail.
    """

    _INPUTS = dict(
        finalWorth=5.0,  # 5 billion USD
        age=52.0,
        category="Technology",
        country="United States",
        gender="M",
        selfMade=1,
    )
    # log1p(5_000) ≈ 8.52  (5 billion → 5000 million, correct scale)
    # log1p(5)     ≈ 1.79  (5 billion passed raw, wrong scale)
    _LOG_WORTH_MIN = 6.0
    _LOG_WORTH_MAX = 14.0

    def test_log_worth_in_training_range(self, loaded_predictor: BillionairesPredictor) -> None:
        row = loaded_predictor._build_row(**self._INPUTS)
        log_worth = float(row["log_worth"].iloc[0])
        assert self._LOG_WORTH_MIN < log_worth < self._LOG_WORTH_MAX, (
            f"log_worth={log_worth:.3f} is outside the training range "
            f"[{self._LOG_WORTH_MIN}, {self._LOG_WORTH_MAX}]. "
            "This indicates the billion→million unit conversion in _build_row "
            "is missing or incorrect (C1)."
        )

    def test_final_worth_stored_as_millions(self, loaded_predictor: BillionairesPredictor) -> None:
        """finalWorth in the built row must be ~5 000 (millions), not 5 (billions)."""
        row = loaded_predictor._build_row(**self._INPUTS)
        fw = float(row["finalWorth"].iloc[0])
        assert fw > 100, (
            f"finalWorth={fw} looks like billions, not millions. "
            "Check the *1000 conversion in _build_row."
        )

    def test_wealth_per_decade_not_tiny(self, loaded_predictor: BillionairesPredictor) -> None:
        """wealth_per_decade must also be on the millions scale."""
        row = loaded_predictor._build_row(**self._INPUTS)
        wpd = float(row["wealth_per_decade"].iloc[0])
        # 5000 millions / 5.2 decades ≈ 961 — well above 1.0
        assert wpd > 10, (
            f"wealth_per_decade={wpd:.3f} is suspiciously small — "
            "suggests finalWorth was not converted to millions."
        )


# ── predict_self_made ──────────────────────────────────────────────────────


class TestPredictSelfMade:
    _COMMON = dict(
        finalWorth=5.0,
        age=52,
        category="Technology",
        country="United States",
        gender="M",
    )

    def test_returns_required_keys(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_self_made(**self._COMMON)
        assert set(result) == {"probability", "prediction", "label"}

    def test_probability_in_unit_interval(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_self_made(**self._COMMON)
        assert 0.0 <= result["probability"] <= 1.0

    def test_prediction_is_binary(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_self_made(**self._COMMON)
        assert result["prediction"] in (0, 1)

    def test_label_matches_prediction(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_self_made(**self._COMMON)
        expected = "Self-Made" if result["prediction"] == 1 else "Inherited"
        assert result["label"] == expected

    def test_deterministic(self, loaded_predictor: BillionairesPredictor) -> None:
        r1 = loaded_predictor.predict_self_made(**self._COMMON)
        r2 = loaded_predictor.predict_self_made(**self._COMMON)
        assert r1["probability"] == r2["probability"]

    def test_unknown_country_does_not_raise(self, loaded_predictor: BillionairesPredictor) -> None:
        """OrdinalEncoder unknown_value=-1 must absorb unseen categories."""
        result = loaded_predictor.predict_self_made(
            finalWorth=5.0,
            age=52,
            category="Technology",
            country="Wakanda",
            gender="M",
        )
        assert "probability" in result


# ── predict_worth ──────────────────────────────────────────────────────────


class TestPredictWorth:
    _COMMON = dict(
        finalWorth=5.0,
        age=52,
        category="Technology",
        country="United States",
        gender="M",
        selfMade=1,
    )

    def test_returns_required_keys(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_worth(**self._COMMON)
        assert set(result) == {
            "log_worth_predicted",
            "worth_billion_usd",
            "selfMade_used",
        }

    def test_log_worth_positive(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_worth(**self._COMMON)
        assert result["log_worth_predicted"] > 0

    def test_worth_billion_usd_positive(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_worth(**self._COMMON)
        assert result["worth_billion_usd"] > 0

    def test_back_transform_consistency(self, loaded_predictor: BillionairesPredictor) -> None:
        """expm1(log_worth_predicted) / 1000 must equal worth_billion_usd."""
        import math

        result = loaded_predictor.predict_worth(**self._COMMON)
        expected = math.expm1(result["log_worth_predicted"]) / 1_000
        assert abs(result["worth_billion_usd"] - expected) < 0.01

    def test_selfmade_used_propagated(self, loaded_predictor: BillionairesPredictor) -> None:
        for sm in (0, 1):
            result = loaded_predictor.predict_worth(
                finalWorth=5.0,
                age=52,
                category="Technology",
                country="United States",
                gender="M",
                selfMade=sm,
            )
            assert result["selfMade_used"] == sm


# ── predict_cluster ────────────────────────────────────────────────────────


class TestPredictCluster:
    _COMMON = dict(
        finalWorth=5.0,
        age=52,
        category="Technology",
        country="United States",
        gender="M",
        selfMade=1,
    )

    def test_returns_required_keys(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_cluster(**self._COMMON)
        assert set(result) == {"cluster", "n_clusters", "silhouette"}

    def test_cluster_id_in_range(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_cluster(**self._COMMON)
        assert 0 <= result["cluster"] < result["n_clusters"]

    def test_n_clusters_matches_model(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_cluster(**self._COMMON)
        assert result["n_clusters"] == loaded_predictor._clusterer.k_


# ── predict_all ────────────────────────────────────────────────────────────


class TestPredictAll:
    _COMMON = dict(
        finalWorth=5.0,
        age=52,
        category="Technology",
        country="United States",
        gender="M",
    )

    def test_returns_all_three_keys(self, loaded_predictor: BillionairesPredictor) -> None:
        result = loaded_predictor.predict_all(**self._COMMON)
        assert set(result) == {"self_made", "worth", "cluster"}

    def test_selfmade_propagated_to_worth(self, loaded_predictor: BillionairesPredictor) -> None:
        """selfMade used by regressor must equal the classifier's prediction."""
        result = loaded_predictor.predict_all(**self._COMMON)
        assert result["worth"]["selfMade_used"] == result["self_made"]["prediction"]

    def test_selfmade_propagated_to_cluster(self, loaded_predictor: BillionairesPredictor) -> None:
        """Cluster model uses the same selfMade as the classifier output."""
        # We can't directly assert this from predict_all's return, but we can
        # verify that calling predict_cluster with the same selfMade produces
        # the same cluster ID.
        result = loaded_predictor.predict_all(**self._COMMON)
        cluster_direct = loaded_predictor.predict_cluster(
            selfMade=result["self_made"]["prediction"], **self._COMMON
        )
        assert result["cluster"]["cluster"] == cluster_direct["cluster"]

    def test_deterministic_across_calls(self, loaded_predictor: BillionairesPredictor) -> None:
        r1 = loaded_predictor.predict_all(**self._COMMON)
        r2 = loaded_predictor.predict_all(**self._COMMON)
        assert r1["self_made"]["probability"] == r2["self_made"]["probability"]
        assert r1["worth"]["log_worth_predicted"] == r2["worth"]["log_worth_predicted"]
        assert r1["cluster"]["cluster"] == r2["cluster"]["cluster"]
