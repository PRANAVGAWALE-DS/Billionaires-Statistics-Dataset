"""
pipeline.py
-----------
Billionaires Statistics — end-to-end training + serialisation pipeline.

Orchestrates:
    load → validate → clean → build_features → stratified split →
    FeatureEncoder (train-only fit) → SelfMadeClassifier (tune + fit) →
    WorthRegressor (tune + fit) → BillionaireClusterer (fit) →
    evaluate all → persist all → print summary.

Usage
-----
# Full run — 40 Optuna trials per model (default)
python pipeline.py

# Fast dev mode — skip HPO, XGBoost defaults, fixed k
python pipeline.py --skip-hpo --cluster-k 4

# Override HPO budget and paths
python pipeline.py --n-trials 20 --seed 0 --out-dir data/processed

Serialised outputs  →  data/processed/
--------------------------------------------
classifier.joblib       SelfMadeClassifier wrapper  (predict / predict_proba)
classifier_xgb.json     XGBoost native              (portable, pre-ONNX path)
regressor.joblib        WorthRegressor wrapper       (predict, exponentiate=True)
regressor_xgb.json      XGBoost native
clusterer.joblib        BillionaireClusterer         (scaler + PCA + KMeans)
feature_encoder.joblib  FeatureEncoder               (fitted OrdinalEncoder)
feature_cols.json       {"clf": [...], "reg": [...], "cluster": [...]}
metrics.json            all evaluation metrics       (JSON-serialisable)
"""

from __future__ import annotations

# Must be set before sklearn/numpy import to prevent OpenMP deadlock on Windows
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from billionaires.data.loader import clean, load_raw, validate_schema
from billionaires.features.engineer import (
    FeatureEncoder,
    build_features,
    get_clf_features,
    get_cluster_features,
    get_reg_features,
)
from billionaires.models.classifier import SelfMadeClassifier
from billionaires.models.clusterer import BillionaireClusterer
from billionaires.models.evaluate import (
    evaluate_classifier,
    evaluate_clusters,
    evaluate_regressor,
    metrics_table,
)
from billionaires.models.regressor import WorthRegressor

# ── Pipeline defaults ──────────────────────────────────────────────────────
_DATA_PATH = Path("data/raw/Billionaires Statistics Dataset.csv")
_OUT_DIR = Path("data/processed")
_N_TRIALS = 40
_CV_FOLDS = 5
_SEED = 42
_VAL_SIZE = 0.15  # fraction of full dataset
_TEST_SIZE = 0.15  # fraction of full dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── Internal helpers ───────────────────────────────────────────────────────


def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy / pandas types to JSON-serialisable Python.

    Handles:
        np.ndarray  → list (via .tolist())
        np.integer  → int
        np.floating → float
        dict / list → recurse
        str / int / float / None → unchanged
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _stratified_split(
    df: pd.DataFrame,
    *,
    val_size: float,
    test_size: float,
    stratify_col: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames via two stratified splits.

    Both splits stratify on *stratify_col* so class balance is preserved
    across all three partitions.  Target proportions relative to the full
    dataset are guaranteed to match *val_size* and *test_size* exactly.

    Parameters
    ----------
    df           : Full feature-engineered DataFrame (pre-encoding).
    val_size     : Desired val fraction of the full dataset (e.g. 0.15).
    test_size    : Desired test fraction of the full dataset (e.g. 0.15).
    stratify_col : Column used for stratification (``"selfMade"``).
    seed         : Random state passed to both splits.

    Returns
    -------
    (df_train, df_val, df_test)
    """
    # Step 1 — carve out the test set
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=seed,
    )
    # Step 2 — carve val from the remainder
    # val_fraction_of_trainval = val_size / (1 - test_size)
    val_frac = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_frac,
        stratify=df_trainval[stratify_col],
        random_state=seed,
    )
    logger.info(
        "Split → train: %d  val: %d  test: %d  (stratified on '%s')",
        len(df_train),
        len(df_val),
        len(df_test),
        stratify_col,
    )
    return df_train, df_val, df_test


# ── Model runners ──────────────────────────────────────────────────────────


def _run_classifier(
    df_train_enc: pd.DataFrame,
    df_val_enc: pd.DataFrame,
    df_test_enc: pd.DataFrame,
    *,
    n_trials: int,
    cv_folds: int,
    seed: int,
    skip_hpo: bool,
) -> tuple[SelfMadeClassifier, dict[str, Any]]:
    """Tune → fit → evaluate the self-made classifier.

    Returns
    -------
    (SelfMadeClassifier, {"val": val_metrics, "test": test_metrics})
    """
    feat_cols = get_clf_features()
    target_col = "selfMade"

    X_train = df_train_enc[feat_cols].to_numpy()
    y_train = df_train_enc[target_col].to_numpy()
    X_val = df_val_enc[feat_cols].to_numpy()
    y_val = df_val_enc[target_col].to_numpy()
    X_test = df_test_enc[feat_cols].to_numpy()
    y_test = df_test_enc[target_col].to_numpy()

    clf = SelfMadeClassifier(n_trials=n_trials, cv_folds=cv_folds, seed=seed)

    if not skip_hpo:
        logger.info("Classifier HPO — %d trials × %d-fold CV ...", n_trials, cv_folds)
        clf.tune(X_train, y_train)
    else:
        logger.info("Classifier HPO skipped (--skip-hpo)")

    logger.info("Fitting classifier ...")
    clf.fit(X_train, y_train, X_val, y_val)

    val_metrics = evaluate_classifier(
        y_val,
        clf.predict(X_val),
        clf.predict_proba(X_val)[:, 1],
        split="val",
    )
    test_metrics = evaluate_classifier(
        y_test,
        clf.predict(X_test),
        clf.predict_proba(X_test)[:, 1],
        split="test",
    )
    return clf, {"val": val_metrics, "test": test_metrics}


def _run_regressor(
    df_train_enc: pd.DataFrame,
    df_val_enc: pd.DataFrame,
    df_test_enc: pd.DataFrame,
    *,
    n_trials: int,
    cv_folds: int,
    seed: int,
    skip_hpo: bool,
) -> tuple[WorthRegressor, dict[str, Any]]:
    """Tune → fit → evaluate the net-worth regressor.

    Target is ``log_worth`` (log1p scale).  Dollar-scale MAPE is also
    computed (``exponentiated=True``) and stored in the metrics dict.

    Returns
    -------
    (WorthRegressor, {"val": val_metrics, "test": test_metrics})
    """
    feat_cols = get_reg_features()
    target_col = "log_worth"

    X_train = df_train_enc[feat_cols].to_numpy()
    y_train = df_train_enc[target_col].to_numpy()
    X_val = df_val_enc[feat_cols].to_numpy()
    y_val = df_val_enc[target_col].to_numpy()
    X_test = df_test_enc[feat_cols].to_numpy()
    y_test = df_test_enc[target_col].to_numpy()

    reg = WorthRegressor(n_trials=n_trials, cv_folds=cv_folds, seed=seed)

    if not skip_hpo:
        logger.info("Regressor HPO — %d trials × %d-fold CV ...", n_trials, cv_folds)
        reg.tune(X_train, y_train)
    else:
        logger.info("Regressor HPO skipped (--skip-hpo)")

    logger.info("Fitting regressor ...")
    reg.fit(X_train, y_train, X_val, y_val)

    val_metrics = evaluate_regressor(
        y_val,
        reg.predict(X_val),
        exponentiated=True,
        split="val",
    )
    test_metrics = evaluate_regressor(
        y_test,
        reg.predict(X_test),
        exponentiated=True,
        split="test",
    )
    return reg, {"val": val_metrics, "test": test_metrics}


def _run_clusterer(
    df_full_enc: pd.DataFrame,
    *,
    cluster_k: int | None,
    seed: int,
) -> tuple[BillionaireClusterer, dict[str, float]]:
    """Fit the clusterer on the full encoded dataset and evaluate quality.

    The clusterer is intentionally fit on the full dataset rather than only
    the training split — K-Means here is used for exploratory segmentation,
    not prediction.  Fitting on all 2,640 rows produces more stable and
    interpretable cluster centroids for profiling and visualisation.

    Parameters
    ----------
    df_full_enc : Full dataset after FeatureEncoder.transform (all rows).
    cluster_k   : Fixed k (skips elbow search) or None for auto-selection.
    seed        : Random seed.

    Returns
    -------
    (BillionaireClusterer, cluster_metrics_dict)
    """
    feat_cols = get_cluster_features()
    # .dropna() mirrors the notebook guard — KMeans rejects NaN rows
    X_cluster = df_full_enc[feat_cols].dropna().to_numpy()
    nan_count = int(np.isnan(X_cluster).sum())
    logger.info(
        "Cluster input: %d rows × %d cols, %d NaN values",
        X_cluster.shape[0],
        X_cluster.shape[1],
        nan_count,
    )
    if nan_count > 0:
        raise ValueError(
            f"X_cluster still contains {nan_count} NaN values after dropna(). "
            "Check feature engineering for inf / NaN sources."
        )

    clusterer = BillionaireClusterer(k_range=(2, 10), seed=seed)
    logger.info(
        "Fitting clusterer%s ...",
        f" (fixed k={cluster_k})" if cluster_k is not None else " (auto elbow search)",
    )
    # k is a positional default-None arg: .fit(X) for auto, .fit(X, k) for fixed
    clusterer.fit(X_cluster, cluster_k)

    # evaluate_clusters expects already-scaled features
    X_scaled = clusterer.scaler_.transform(X_cluster)
    cluster_metrics = evaluate_clusters(X_scaled, clusterer.labels_, split="full")
    return clusterer, cluster_metrics


# ── Serialisation ──────────────────────────────────────────────────────────


def _serialize(
    out_dir: Path,
    *,
    clf: SelfMadeClassifier,
    reg: WorthRegressor,
    clusterer: BillionaireClusterer,
    enc: FeatureEncoder,
    clf_metrics: dict[str, Any],
    reg_metrics: dict[str, Any],
    cluster_metrics: dict[str, float],
) -> None:
    """Persist all model artifacts and evaluation metrics to *out_dir*.

    Two serialisation strategies are used in parallel:

    joblib (.joblib)
        Full Python wrapper objects.  These preserve all Python-level logic
        (e.g. WorthRegressor.predict(exponentiate=True), BillionaireClusterer
        .cluster_profiles()).  Required for the FastAPI inference layer.

    XGBoost native (.json)
        The underlying xgb.XGBClassifier / xgb.XGBRegressor models saved
        with .save_model().  Framework-portable and the required format for
        ONNX export in Phase D.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── joblib wrappers ─────────────────────────────────────────────────
    joblib.dump(clf, out_dir / "classifier.joblib")
    joblib.dump(reg, out_dir / "regressor.joblib")
    joblib.dump(clusterer, out_dir / "clusterer.joblib")
    joblib.dump(enc, out_dir / "feature_encoder.joblib")
    logger.info("Saved .joblib wrappers → %s", out_dir)

    # ── XGBoost native saves ────────────────────────────────────────────
    clf.model_.save_model(str(out_dir / "classifier_xgb.json"))
    reg.model_.save_model(str(out_dir / "regressor_xgb.json"))
    logger.info("Saved XGBoost native models → %s", out_dir)

    # ── Feature column manifest ─────────────────────────────────────────
    # Stored separately so the inference layer can validate its input schema
    # without loading any model object.
    feature_cols: dict[str, list[str]] = {
        "clf": get_clf_features(),
        "reg": get_reg_features(),
        "cluster": get_cluster_features(),
    }
    (out_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))

    # ── Metrics ─────────────────────────────────────────────────────────
    # _json_safe converts np.ndarray (confusion_matrix) → list,
    # np.floating → float, etc.  classification_report strings pass through.
    all_metrics = {
        "classifier": _json_safe(clf_metrics),
        "regressor": _json_safe(reg_metrics),
        "clusterer": _json_safe(cluster_metrics),
    }
    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    logger.info("Saved feature_cols.json + metrics.json → %s", out_dir)

    # ── Manifest listing ────────────────────────────────────────────────
    saved = sorted(out_dir.iterdir())
    logger.info("data/processed/ contents (%d files):", len(saved))
    for f in saved:
        size_kb = f.stat().st_size / 1024
        logger.info("  %-32s  %6.1f KB", f.name, size_kb)


# ── Summary printer ────────────────────────────────────────────────────────


def _print_summary(
    clf_metrics: dict[str, Any],
    reg_metrics: dict[str, Any],
    cluster_metrics: dict[str, float],
    elapsed: float,
) -> None:
    """Print a formatted results summary to stdout."""
    bar = "═" * 62
    thin = "─" * 62

    print(f"\n{bar}")
    print("  BILLIONAIRES PIPELINE — COMPLETE")
    print(bar)

    print("\n  ① Self-Made Classifier  (XGBoost + Optuna HPO)")
    print(thin)
    clf_table = metrics_table({"val": clf_metrics["val"], "test": clf_metrics["test"]})
    # drop non-numeric keys (classification_report, confusion_matrix) already
    # handled by metrics_table — just print what's there
    print(clf_table.to_string())

    print("\n  ② Net-Worth Regressor  (log scale + dollar-scale MAPE)")
    print(thin)
    reg_table = metrics_table({"val": reg_metrics["val"], "test": reg_metrics["test"]})
    print(reg_table.to_string())

    print("\n  ③ Wealth Segment Clusterer  (K-Means, full dataset)")
    print(thin)
    print(
        f"  k = {int(cluster_metrics['n_clusters'])}  │  "
        f"silhouette = {cluster_metrics['silhouette']:.4f}  │  "
        f"davies-bouldin = {cluster_metrics['davies_bouldin']:.4f}"
    )

    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print(bar + "\n")


# ── Main orchestrator ──────────────────────────────────────────────────────


def run_pipeline(
    data_path: Path = _DATA_PATH,
    out_dir: Path = _OUT_DIR,
    n_trials: int = _N_TRIALS,
    cv_folds: int = _CV_FOLDS,
    seed: int = _SEED,
    val_size: float = _VAL_SIZE,
    test_size: float = _TEST_SIZE,
    skip_hpo: bool = False,
    cluster_k: int | None = None,
) -> dict[str, Any]:
    """Execute the complete billionaires ML pipeline end-to-end.

    This function is importable for programmatic use (e.g. notebooks,
    tests) as well as callable from the CLI via ``__main__``.

    Parameters
    ----------
    data_path  : Path to ``Billionaires Statistics Dataset.csv``.
    out_dir    : Directory for all serialised outputs.
    n_trials   : Optuna HPO trials per model (classifier + regressor each).
    cv_folds   : CV folds during HPO.
    seed       : Global random seed (applied to splits, HPO, models).
    val_size   : Validation fraction of the full dataset.
    test_size  : Test fraction of the full dataset.
    skip_hpo   : Skip Optuna; use XGBoost defaults.  Useful for fast
                 iteration / CI runs.
    cluster_k  : Fix the K-Means cluster count.  None = auto via silhouette.

    Returns
    -------
    dict
        {"clf": {"val": ..., "test": ...},
         "reg": {"val": ..., "test": ...},
         "cluster": {...}}
    """
    t0 = time.perf_counter()
    logger.info("═" * 50)
    logger.info("Billionaires ML Pipeline — starting")
    logger.info(
        "seed=%d  n_trials=%d  cv_folds=%d  skip_hpo=%s",
        seed,
        n_trials,
        cv_folds,
        skip_hpo,
    )
    logger.info("═" * 50)

    # ── 1. Load, validate, clean ──────────────────────────────────────────
    logger.info("Step 1/7 — Load + validate + clean")
    df_raw = load_raw(data_path)
    validate_schema(df_raw)
    df_clean = clean(df_raw)

    # ── 2. Base feature engineering — NO encoding (split-safe) ───────────
    logger.info("Step 2/7 — Build base features (encode=False)")
    df_feat = build_features(df_clean, encode=False)

    # ── 3. Stratified 70 / 15 / 15 split on selfMade ─────────────────────
    logger.info(
        "Step 3/7 — Stratified split (%.0f/%.0f/%.0f)",
        (1 - val_size - test_size) * 100,
        val_size * 100,
        test_size * 100,
    )
    df_train, df_val, df_test = _stratified_split(
        df_feat,
        val_size=val_size,
        test_size=test_size,
        stratify_col="selfMade",
        seed=seed,
    )

    # ── 4. FeatureEncoder — fit on train only, transform all splits ───────
    logger.info("Step 4/7 — Fit FeatureEncoder on training split only")
    enc = FeatureEncoder()
    df_train_enc = enc.fit_transform(df_train)
    df_val_enc = enc.transform(df_val)
    df_test_enc = enc.transform(df_test)
    # Full-dataset encoding for the clusterer (unsupervised — no leakage risk)
    df_full_enc = enc.transform(df_feat)

    # ── 5. Self-Made Classifier ───────────────────────────────────────────
    logger.info("Step 5/7 — SelfMadeClassifier")
    clf, clf_metrics = _run_classifier(
        df_train_enc,
        df_val_enc,
        df_test_enc,
        n_trials=n_trials,
        cv_folds=cv_folds,
        seed=seed,
        skip_hpo=skip_hpo,
    )

    # ── 6. Net-Worth Regressor ────────────────────────────────────────────
    logger.info("Step 6/7 — WorthRegressor")
    reg, reg_metrics = _run_regressor(
        df_train_enc,
        df_val_enc,
        df_test_enc,
        n_trials=n_trials,
        cv_folds=cv_folds,
        seed=seed,
        skip_hpo=skip_hpo,
    )

    # ── 7. Clusterer ──────────────────────────────────────────────────────
    logger.info("Step 7/7 — BillionaireClusterer")
    clusterer, cluster_metrics = _run_clusterer(
        df_full_enc,
        cluster_k=cluster_k,
        seed=seed,
    )

    # ── Serialise ─────────────────────────────────────────────────────────
    logger.info("Serialising artifacts ...")
    _serialize(
        out_dir,
        clf=clf,
        reg=reg,
        clusterer=clusterer,
        enc=enc,
        clf_metrics=clf_metrics,
        reg_metrics=reg_metrics,
        cluster_metrics=cluster_metrics,
    )

    results: dict[str, Any] = {
        "clf": clf_metrics,
        "reg": reg_metrics,
        "cluster": cluster_metrics,
    }
    elapsed = time.perf_counter() - t0
    _print_summary(clf_metrics, reg_metrics, cluster_metrics, elapsed)
    return results


# ── CLI entry point ────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="Billionaires Statistics — end-to-end ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=_DATA_PATH,
        metavar="PATH",
        help="Path to 'Billionaires Statistics Dataset.csv'.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_OUT_DIR,
        metavar="DIR",
        help="Output directory for serialised artifacts.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=_N_TRIALS,
        metavar="N",
        help="Optuna HPO trials per model (classifier + regressor each).",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=_CV_FOLDS,
        metavar="K",
        help="Cross-validation folds during HPO.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=_SEED,
        help="Global random seed.",
    )
    p.add_argument(
        "--val-size",
        type=float,
        default=_VAL_SIZE,
        metavar="F",
        help="Validation fraction of the full dataset (e.g. 0.15).",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=_TEST_SIZE,
        metavar="F",
        help="Test fraction of the full dataset (e.g. 0.15).",
    )
    p.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Skip Optuna HPO — use XGBoost defaults.  Fast dev / CI mode.",
    )
    p.add_argument(
        "--cluster-k",
        type=int,
        default=None,
        metavar="K",
        help="Fix K-Means cluster count (skips elbow search).  Omit = auto.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        data_path=args.data_path,
        out_dir=args.out_dir,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        skip_hpo=args.skip_hpo,
        cluster_k=args.cluster_k,
    )
    sys.exit(0)
