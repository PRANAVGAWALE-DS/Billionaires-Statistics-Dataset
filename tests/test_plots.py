"""
tests/test_plots.py
-------------------
Smoke tests for billionaires.viz.plots.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure

from billionaires.viz.plots import (
    actual_vs_predicted,
    age_distribution,
    choropleth_count,
    choropleth_wealth,
    confusion_matrix_plot,
    correlation_heatmap,
    gender_breakdown,
    lorenz_curve,
    self_made_comparison,
    top_categories,
    wealth_distribution,
)


def _assert_saved_matplotlib(fig: Figure, output_path) -> None:
    assert isinstance(fig, Figure)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    plt.close(fig)


def test_wealth_distribution_returns_and_saves_figure(
    sample_df_feat: pd.DataFrame, tmp_path
) -> None:
    output_path = tmp_path / "wealth.png"

    fig = wealth_distribution(sample_df_feat, save_path=output_path)

    _assert_saved_matplotlib(fig, output_path)


def test_age_distribution_returns_and_saves_figure(sample_df_feat: pd.DataFrame, tmp_path) -> None:
    output_path = tmp_path / "age.png"

    fig = age_distribution(sample_df_feat, save_path=output_path)

    _assert_saved_matplotlib(fig, output_path)


def test_lorenz_curve_returns_and_saves_figure(sample_df_feat: pd.DataFrame, tmp_path) -> None:
    output_path = tmp_path / "lorenz.png"

    fig = lorenz_curve(sample_df_feat, save_path=output_path)

    _assert_saved_matplotlib(fig, output_path)


def test_correlation_heatmap_returns_and_saves_figure(
    sample_df_feat: pd.DataFrame, tmp_path
) -> None:
    output_path = tmp_path / "corr.png"

    fig = correlation_heatmap(sample_df_feat, save_path=output_path)

    _assert_saved_matplotlib(fig, output_path)


def test_confusion_matrix_plot_returns_and_saves_figure(tmp_path) -> None:
    output_path = tmp_path / "cm.png"

    fig = confusion_matrix_plot(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        save_path=output_path,
    )

    _assert_saved_matplotlib(fig, output_path)


def test_actual_vs_predicted_returns_and_saves_figure(tmp_path) -> None:
    output_path = tmp_path / "actual-vs-pred.png"

    fig = actual_vs_predicted(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.2, 1.8, 3.1]),
        r2=0.75,
        save_path=output_path,
    )

    _assert_saved_matplotlib(fig, output_path)


def test_top_categories_returns_plotly_bar(sample_df_feat: pd.DataFrame) -> None:
    fig = top_categories(sample_df_feat)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_gender_breakdown_returns_two_traces(sample_df_feat: pd.DataFrame) -> None:
    fig = gender_breakdown(sample_df_feat)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_self_made_comparison_returns_grouped_bars(sample_df_feat: pd.DataFrame) -> None:
    fig = self_made_comparison(sample_df_feat)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_choropleth_count_returns_map(sample_df_feat: pd.DataFrame) -> None:
    fig = choropleth_count(sample_df_feat)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "choropleth"


def test_choropleth_wealth_returns_map(sample_df_feat: pd.DataFrame) -> None:
    fig = choropleth_wealth(sample_df_feat)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "choropleth"
