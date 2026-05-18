"""
billionaires.models.clusterer
------------------------------
K-Means clustering on the billionaire feature space.

Responsibilities
----------------
- Scale features with :class:`~sklearn.preprocessing.StandardScaler`
- Elbow search across ``k_range`` with inertia + silhouette scoring
- Fit final K-Means at the chosen k
- PCA 2D projection for scatter visualisation
- Per-cluster profile table for interpretation

Usage
-----
::

    from billionaires.features.engineer import build_features, get_cluster_features
    from billionaires.models.clusterer import BillionaireClusterer

    df_feat = build_features(df_clean)
    feature_cols = get_cluster_features()
    X = df_feat[feature_cols].to_numpy()

    clusterer = BillionaireClusterer(k_range=(2, 10), seed=42)
    clusterer.fit(X)                         # auto-selects best k via silhouette

    df_feat["cluster"] = clusterer.labels_
    xy = clusterer.transform_pca(X)          # 2D coords for scatter plot

    print(clusterer.diagnostics())           # inertia + silhouette per k
    print(clusterer.cluster_profiles(df_feat, feature_cols))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class BillionaireClusterer:
    """K-Means clusterer with elbow diagnostics and PCA projection.

    Parameters
    ----------
    k_range : tuple[int, int]
        Inclusive (min_k, max_k) range to test during elbow search.
        Ignored if ``k`` is passed explicitly to :meth:`fit`.
    seed : int
        Random seed for KMeans and PCA.

    Attributes
    ----------
    k_ : int
        Chosen number of clusters (set after :meth:`fit`).
    scaler_ : StandardScaler
        Fitted scaler (set after :meth:`fit`).
    pca_ : PCA
        Fitted 2-component PCA (set after :meth:`fit`).
    model_ : KMeans
        Fitted KMeans model (set after :meth:`fit`).
    labels_ : np.ndarray, shape (n_samples,)
        Cluster assignments for the training data (set after :meth:`fit`).
    inertias_ : dict[int, float]
        Per-k inertia values from elbow search.
    silhouettes_ : dict[int, float]
        Per-k silhouette scores from elbow search.
    """

    k_range: tuple[int, int] = (2, 10)
    silhouette_sample_size: int | None = 500  # subsample for speed; None = full dataset
    seed: int = 42

    # post-fit attributes — not constructor arguments
    k_: int = field(default=0, init=False, repr=False)
    scaler_: StandardScaler | None = field(default=None, init=False, repr=False)
    pca_: PCA | None = field(default=None, init=False, repr=False)
    model_: KMeans | None = field(default=None, init=False, repr=False)
    labels_: np.ndarray | None = field(default=None, init=False, repr=False)
    inertias_: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    silhouettes_: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    # ── Public API ────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, k: int | None = None) -> "BillionaireClusterer":
        """Scale → (optionally) elbow search → fit final K-Means → fit PCA.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Raw (unscaled) feature matrix.
        k : int | None
            If provided, skip the elbow search and use this k directly.
            If None (default), select k automatically via max silhouette.

        Returns
        -------
        self
        """
        # 1. Scale — K-Means is distance-based; scaling is mandatory.
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # 2. Select k.
        if k is None:
            k = self._elbow_search(X_scaled)
        self.k_ = k

        # 3. Fit final model.
        self.model_ = KMeans(
            n_clusters=k,
            random_state=self.seed,
            n_init=1,  # avoid OpenMP thread-pool deadlock on Windows
            algorithm="lloyd",
        )
        self.labels_ = self.model_.fit_predict(X_scaled)

        # 4. Fit PCA for 2D visualisation (on scaled data).
        self.pca_ = PCA(n_components=2, random_state=self.seed)
        self.pca_.fit(X_scaled)

        logger.info(
            "BillionaireClusterer fitted — k=%d, inertia=%.2f, explained variance (2 PC) = %.1f%%",
            k,
            float(self.model_.inertia_),
            float(self.pca_.explained_variance_ratio_.sum() * 100),
        )
        return self

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """Scale *X* and project to the fitted 2D PCA space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Raw (unscaled) feature matrix — same feature order as in fit().

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            PCA coordinates (PC1, PC2) for scatter visualisation.
        """
        self._check_fitted()
        X_scaled = self.scaler_.transform(X)
        return self.pca_.transform(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign cluster labels to new (unseen) data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Raw (unscaled) feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Integer cluster labels.
        """
        self._check_fitted()
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def cluster_profiles(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Per-cluster mean and median for each feature.

        Parameters
        ----------
        df : pd.DataFrame
            Must be aligned with the X passed to :meth:`fit` (same row order).
            The ``cluster`` column is added temporarily; it is not required
            to already exist.
        feature_cols : list[str]
            Columns to include in the profile.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns: (feature, stat) where stat ∈ {mean, median}.
            Index is the cluster id (0 … k-1).
        """
        self._check_fitted()
        profile_df = df[feature_cols].copy()
        profile_df["cluster"] = self.labels_
        return profile_df.groupby("cluster")[feature_cols].agg(["mean", "median"]).round(3)

    def diagnostics(self) -> pd.DataFrame:
        """Return a tidy DataFrame of inertia and silhouette per k tested.

        Returns an empty DataFrame if :meth:`fit` was called with an explicit
        k (elbow search skipped).

        Returns
        -------
        pd.DataFrame
            Columns: inertia, silhouette.  Index: k.
        """
        if not self.inertias_:
            return pd.DataFrame(columns=["inertia", "silhouette"])
        return pd.DataFrame(
            {
                "inertia": self.inertias_,
                "silhouette": self.silhouettes_,
            }
        ).rename_axis("k")

    # ── Private ───────────────────────────────────────────────────────────

    def _elbow_search(self, X_scaled: np.ndarray) -> int:
        """Compute inertia + silhouette for each k in k_range.

        Selection criterion: k that maximises the silhouette score.
        Inertia is stored for elbow plotting in the notebook.

        Parameters
        ----------
        X_scaled : np.ndarray
            Already-scaled feature matrix.

        Returns
        -------
        int
            Chosen k.
        """
        k_min, k_max = self.k_range
        for k in range(k_min, k_max + 1):
            km = KMeans(
                n_clusters=k,
                random_state=self.seed,
                n_init=1,
                algorithm="lloyd",
            )
            labels = km.fit_predict(X_scaled)
            self.inertias_[k] = float(km.inertia_)
            self.silhouettes_[k] = float(
                silhouette_score(
                    X_scaled,
                    labels,
                    sample_size=self.silhouette_sample_size,
                    random_state=self.seed,
                )
            )
            logger.debug(
                "k=%d  inertia=%.2f  silhouette=%.4f",
                k,
                self.inertias_[k],
                self.silhouettes_[k],
            )

        best_k = max(self.silhouettes_, key=lambda k: self.silhouettes_[k])
        logger.info(
            "Elbow search complete — best k=%d (silhouette=%.4f)",
            best_k,
            self.silhouettes_[best_k],
        )
        return best_k

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("BillionaireClusterer is not fitted. Call .fit() first.")
