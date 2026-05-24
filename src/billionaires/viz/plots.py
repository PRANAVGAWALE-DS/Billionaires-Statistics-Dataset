"""
billionaires.viz.plots
-----------------------
All visualisation functions for the project.

Design rules
------------
* Every function returns a figure object (``plt.Figure`` or
  ``plotly.graph_objects.Figure``) so notebooks can call ``.show()``
  or ``.write_image()`` as needed.
* No ``plt.show()`` inside functions — the caller decides.
* Static charts use Matplotlib/Seaborn; interactive charts use Plotly.
* Functions accept a ``save_path`` kwarg; if provided the figure is
  written to disk before returning.
* ``shap`` is imported lazily inside :func:`shap_summary` to avoid
  loading a heavy optional dependency at module import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

# ── Shared style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 14})
_BLUE = "#4C72B0"
_ORANGE = "#DD8452"


def _maybe_save(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")


# ── EDA plots ─────────────────────────────────────────────────────────────


def wealth_distribution(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Side-by-side raw vs log(1+x) distribution of ``finalWorth``."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(
        df["finalWorth"],
        bins=60,
        kde=True,
        ax=axes[0],
        color=_BLUE,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].set_title("finalWorth Distribution (raw)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}B"))

    sns.histplot(
        df["log_worth"],
        bins=60,
        kde=True,
        ax=axes[1],
        color=_ORANGE,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].set_title("log(1 + finalWorth) Distribution")
    axes[1].axvline(
        df["log_worth"].mean(),
        color="red",
        ls="--",
        lw=1.8,
        label=f"Mean {df['log_worth'].mean():.2f}",
    )
    axes[1].axvline(
        df["log_worth"].median(),
        color="green",
        ls="--",
        lw=1.8,
        label=f"Median {df['log_worth'].median():.2f}",
    )
    axes[1].legend()

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def age_distribution(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Violin (age × self-made) + box (log_worth × age_group)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Convert selfMade to string for palette compatibility
    df_plot = df.copy()
    df_plot["selfMade"] = df_plot["selfMade"].astype(str)
    sns.violinplot(
        data=df_plot,
        x="selfMade",
        y="age",
        palette={"0": _ORANGE, "1": _BLUE},
        inner="box",
        ax=axes[0],
    )
    axes[0].set_xticklabels(["Inherited (0)", "Self-Made (1)"])
    axes[0].set_title("Age Distribution by Wealth Origin")

    # FIX M1 — use hyphens (-) to match _AGE_LABELS in engineer.py.
    # The original code used en-dashes (–) which never matched any category,
    # silently dropping the four middle age bins from the box plot.
    order = ["<40", "40-54", "55-64", "65-79", "80+"]
    available = [o for o in order if o in df["age_group"].cat.categories]
    sns.boxplot(
        data=df,
        x="age_group",
        y="log_worth",
        palette="muted",
        ax=axes[1],
        order=available,
    )
    axes[1].set_title("Log Net Worth by Age Group")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def top_categories(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart — top 15 categories by median net worth."""
    stats = (
        df.groupby("category")["finalWorth"]
        .agg(median="median", mean="mean", count="size")
        .sort_values("median", ascending=False)
        .head(15)
        .reset_index()
    )
    fig = px.bar(
        stats,
        x="median",
        y="category",
        orientation="h",
        color="median",
        color_continuous_scale="Viridis",
        hover_data={"mean": ":.1f", "count": True},
        title="Top 15 Categories — Median Net Worth (B USD)",
        labels={"median": "Median ($B)", "category": "Category"},
    )
    fig.update_layout(yaxis={"autorange": "reversed"}, coloraxis_showscale=False, height=500)
    return fig


def gender_breakdown(df: pd.DataFrame) -> go.Figure:
    """Side-by-side bar subplots: count and median worth by gender."""
    agg = (
        df.groupby("gender")
        .agg(
            count=("finalWorth", "size"),
            median_worth=("finalWorth", "median"),
            pct_self_made=("selfMade", "mean"),
        )
        .reset_index()
    )
    agg["pct_self_made"] = (agg["pct_self_made"] * 100).round(1)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Count by Gender", "Median Net Worth by Gender ($B)"],
    )
    colors = [_BLUE, _ORANGE]
    for col_idx, col_name in enumerate(["count", "median_worth"], 1):
        fig.add_trace(
            go.Bar(
                x=agg["gender"],
                y=agg[col_name],
                text=agg[col_name].round(1),
                textposition="outside",
                marker_color=colors,
            ),
            row=1,
            col=col_idx,
        )
    fig.update_layout(title="Gender Analysis", showlegend=False, height=420)
    return fig


def self_made_comparison(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: mean & median net worth for self-made vs inherited."""
    agg = df.groupby("selfMade")["finalWorth"].agg(["mean", "median"]).reset_index()
    agg["label"] = agg["selfMade"].map({0: "Inherited", 1: "Self-Made"})

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Mean", x=agg["label"], y=agg["mean"], marker_color=_BLUE))
    fig.add_trace(go.Bar(name="Median", x=agg["label"], y=agg["median"], marker_color=_ORANGE))
    fig.update_layout(
        barmode="group",
        title="Net Worth Comparison — Self-Made vs Inherited",
        yaxis_title="Net Worth ($B)",
        height=420,
    )
    return fig


def lorenz_curve(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Lorenz curve + Gini coefficient annotation."""
    values = np.sort(df["finalWorth"].values)
    cum_w = np.cumsum(values) / values.sum()
    cum_pop = np.linspace(0, 1, len(values))

    n = len(values)
    index = np.arange(1, n + 1)
    gini = float((2 * (index * values).sum() / (n * values.sum())) - (n + 1) / n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cum_pop, cum_w, lw=2.5, color=_BLUE, label=f"Lorenz Curve  (Gini = {gini:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Equality")
    ax.fill_between(cum_pop, cum_pop, cum_w, alpha=0.15, color=_BLUE)
    ax.set_xlabel("Cumulative population share")
    ax.set_ylabel("Cumulative wealth share")
    ax.set_title("Lorenz Curve — Wealth Inequality Among Billionaires")
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def choropleth_count(df: pd.DataFrame) -> go.Figure:
    """World map coloured by number of billionaires per country."""
    agg = (
        df.groupby("country")
        .agg(
            count=("finalWorth", "size"),
            total=("finalWorth", "sum"),
            median=("finalWorth", "median"),
        )
        .reset_index()
    )
    fig = px.choropleth(
        agg,
        locations="country",
        locationmode="country names",
        color="count",
        hover_name="country",
        hover_data={"total": ":,.0f", "median": ":,.1f"},
        color_continuous_scale="YlOrRd",
        title="🌍 Number of Billionaires by Country",
    )
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True), height=520)
    return fig


def choropleth_wealth(df: pd.DataFrame) -> go.Figure:
    """World map coloured by total billionaire wealth per country."""
    agg = (
        df.groupby("country")["finalWorth"]
        .sum()
        .reset_index()
        .rename(columns={"finalWorth": "total_worth"})
    )
    fig = px.choropleth(
        agg,
        locations="country",
        locationmode="country names",
        color="total_worth",
        hover_name="country",
        color_continuous_scale="Blues",
        title="💰 Total Billionaire Wealth by Country ($B USD)",
    )
    fig.update_layout(geo=dict(showframe=False), height=520)
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Lower-triangle correlation heatmap for numeric columns."""
    drop_cols = {"birthYear", "rank"}
    num_df = df.select_dtypes(include=np.number).drop(
        columns=[c for c in drop_cols if c in df.columns], errors="ignore"
    )
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation Matrix — Numeric Features")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ── Model evaluation plots ────────────────────────────────────────────────


def confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Annotated confusion matrix heatmap."""
    labels = labels or ["Inherited", "Self-Made"]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Self-Made Classifier")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Scatter of actual vs predicted log(1+finalWorth)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.45, s=20, color=_BLUE)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5)
    ax.set_xlabel("Actual log(1 + finalWorth)")
    ax.set_ylabel("Predicted log(1 + finalWorth)")
    ax.set_title(f"Regressor — Actual vs Predicted  (R² = {r2:.3f})")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def shap_summary(
    model,
    X: np.ndarray,
    feature_names: list[str],
    plot_type: str = "dot",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """SHAP beeswarm (dot) or bar summary plot.

    Parameters
    ----------
    model        : fitted XGBoost model
    X            : feature matrix (same scale as training)
    feature_names: column names matching columns of X
    plot_type    : ``"dot"`` (beeswarm) or ``"bar"``
    save_path    : optional output path

    Returns
    -------
    plt.Figure

    Notes
    -----
    ``shap`` is imported lazily to avoid loading a heavy optional
    dependency for callers that never use SHAP explainability.
    """
    # FIX M7 — deferred import: shap is heavy and optional.
    # Importing at module level would force every consumer of plots.py
    # (including lightweight inference containers) to install shap.
    import shap  # noqa: PLC0415

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    shap.summary_plot(shap_vals, X, feature_names=feature_names, plot_type=plot_type, show=False)
    fig = plt.gcf()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
