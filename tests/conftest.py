"""
tests/conftest.py
-----------------
Shared pytest fixtures.  All fixtures use scope="session" so the
synthetic DataFrame and clean copy are built once per test run.

Thread-count env vars are set here (before any sklearn/numpy import)
to prevent OpenMP deadlocks on Windows and in CI.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest

from billionaires.data.loader import clean


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """120-row synthetic DataFrame matching the real dataset's required schema.

    Columns
    -------
    Includes every column in ``REQUIRED_COLUMNS`` (personName, finalWorth,
    category, country, age, gender, selfMade) plus ``birthYear`` (which
    loader.clean() coerces to numeric).

    selfMade is seeded at 69 / 31 split to match the real class balance
    and ensure both classes are present for stratified splits.
    """
    rng = np.random.default_rng(42)
    n = 120
    return pd.DataFrame(
        {
            "personName": [f"Person {i}" for i in range(n)],
            "finalWorth": rng.uniform(1.0, 200.0, n),
            "category": rng.choice(
                ["Technology", "Finance", "Fashion & Retail", "Manufacturing"], n
            ),
            "country": rng.choice(
                ["United States", "China", "India", "Germany", "Brazil"], n
            ),
            "age": rng.integers(35, 90, n).astype(float),
            "gender": rng.choice(["M", "F"], n),
            "selfMade": rng.choice([True, False], n, p=[0.69, 0.31]),
            "birthYear": rng.integers(1940, 1990, n).astype(float),
        }
    )


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned version of raw_df produced by billionaires.data.loader.clean()."""
    return clean(raw_df)
