"""
billionaires.data.loader
------------------------
Responsibilities
    load_raw        : read CSV → raw DataFrame
    validate_schema : assert required columns exist
    clean           : impute, coerce, deduplicate
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that MUST be present for the pipeline to work
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"personName", "finalWorth", "category", "country", "age", "gender", "selfMade"}
)

# Columns to coerce to numeric during cleaning.
# NOTE — birthYear is coerced here for data-integrity reasons (mixed dtype
# in the raw CSV) but is NOT used as a model feature.  Do not add it to any
# feature list in engineer.py without revisiting the leakage implications.
NUMERIC_COLUMNS: list[str] = ["finalWorth", "age", "birthYear"]


# ── Public API ─────────────────────────────────────────────────────────────


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV from *path* and return a raw, unmodified DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to ``Billionaires Statistics Dataset.csv``.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Place the CSV in data/raw/ and update DATA_PATH in the notebook."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns from '%s'", *df.shape, path.name)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if any required column is missing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned DataFrame to validate.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}\n"
            f"Available columns: {sorted(df.columns.tolist())}"
        )
    logger.info("Schema validation passed ✓")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of *df*.

    Steps
    -----
    1. Coerce numeric columns (``finalWorth``, ``age``, ``birthYear``).
    2. Drop rows where ``finalWorth`` is null — it is the analysis target.
    3. Impute ``age`` with the **per-category median** (falls back to
       the global median for categories that have no age data at all).
    4. Cast ``selfMade`` to ``int`` (0 / 1).
    5. Strip leading/trailing whitespace from string columns.
    6. Drop exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame produced by :func:`load_raw`.

    Returns
    -------
    pd.DataFrame
        A cleaned, independent copy.
    """
    df = df.copy()

    # ── 1. Numeric coercion ──────────────────────────────────────────────
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 2. Drop missing target ───────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["finalWorth"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with null finalWorth", dropped)

    # ── 3. Smart age imputation ──────────────────────────────────────────
    if "age" in df.columns:
        df["age"] = df.groupby("category")["age"].transform(
            lambda x: x.fillna(x.median())
        )
        global_median_age = df["age"].median()
        df["age"] = df["age"].fillna(global_median_age)
        logger.info("Age imputation complete (per-category median)")

    # ── 4. selfMade → int ────────────────────────────────────────────────
    if "selfMade" in df.columns:
        df["selfMade"] = df["selfMade"].astype(bool).astype(int)

    # ── 5. Strip string whitespace ───────────────────────────────────────
    # include=["object", "string"] covers both the legacy object dtype
    # (pandas 2) and the explicit StringDtype (pandas 3).  Using only
    # "object" triggers a Pandas4Warning in pandas 2.x because pandas 3
    # separates string from object and the implicit inclusion is deprecated.
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # ── 6. Deduplicate ───────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    dupes = before - len(df)
    if dupes:
        logger.info("Removed %d duplicate rows", dupes)

    logger.info("Cleaning complete → %d rows × %d columns", *df.shape)
    return df
