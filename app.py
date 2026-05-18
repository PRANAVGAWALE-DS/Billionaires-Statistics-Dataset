"""
app.py
------
Streamlit application for the Billionaires Statistics ML pipeline.

Tabs
----
1. What-If Simulator  — live predictions from all three models
2. Dataset Explorer   — interactive EDA: KPIs, distributions,
                        geography, inequality, top billionaires

Run locally
-----------
    streamlit run app.py

Deploy to Hugging Face Spaces
------------------------------
    Commit this file + src/ + api/ + data/ to the Space repo.
    HF reads requirements.txt automatically.
"""

from __future__ import annotations

# Thread guards — must be set before any sklearn / numpy import
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

# Make src/ and project root importable without pip install -e .
# (works both locally and on HF Spaces)
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Billionaires Statistics — ML Pipeline",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Shared constants ───────────────────────────────────────────────────────

_DATA_PATH = _ROOT / "data" / "raw" / "Billionaires Statistics Dataset.csv"
_PROCESSED = _ROOT / "data" / "processed"

_CATEGORIES = [
    "Technology",
    "Finance",
    "Fashion & Retail",
    "Real Estate",
    "Diversified",
    "Manufacturing",
    "Food & Beverage",
    "Media & Entertainment",
    "Automotive",
    "Healthcare",
    "Metals & Mining",
    "Energy",
    "Telecom",
    "Sports",
    "Construction & Engineering",
    "Logistics",
]

_COUNTRIES = [
    "United States",
    "China",
    "India",
    "Germany",
    "Russia",
    "United Kingdom",
    "France",
    "Brazil",
    "Canada",
    "Australia",
    "Hong Kong",
    "Switzerland",
    "Singapore",
    "Japan",
    "South Korea",
    "Italy",
    "Sweden",
    "Netherlands",
    "Israel",
    "Saudi Arabia",
    "United Arab Emirates",
    "Taiwan",
    "Indonesia",
    "Mexico",
    "South Africa",
    "Turkey",
    "Spain",
    "Norway",
    "Thailand",
    "Nigeria",
]

_CLUSTER_LABELS = {
    0: "Emerging Wealth",
    1: "Mid-Tier Magnate",
    2: "Upper Echelon",
    3: "Ultra-High Net Worth",
}

_PALETTE = px.colors.qualitative.Bold


# ── Cached resource loaders ────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading ML models …")
def _load_predictor():
    from api.predictor import BillionairesPredictor

    p = BillionairesPredictor(processed_dir=_PROCESSED)
    p.load()
    return p


@st.cache_data(show_spinner="Loading dataset …")
def _load_data() -> pd.DataFrame:
    from billionaires.data.loader import clean, load_raw
    from billionaires.features.engineer import build_features

    df = build_features(clean(load_raw(_DATA_PATH)), encode=True)
    return df


# ── Helper functions ───────────────────────────────────────────────────────


def _gini(values: np.ndarray) -> float:
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def _lorenz(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(np.abs(values))
    cumsum = np.cumsum(v)
    return (
        np.linspace(0, 1, len(v) + 1),
        np.concatenate([[0], cumsum / cumsum[-1]]),
    )


def _probability_gauge(prob: float, title: str) -> go.Figure:
    color = "#4CAF50" if prob >= 0.6 else ("#FF9800" if prob >= 0.35 else "#F44336")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35], "color": "#FFEBEE"},
                    {"range": [35, 60], "color": "#FFF3E0"},
                    {"range": [60, 100], "color": "#E8F5E9"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ── Tab 1: What-If Simulator ───────────────────────────────────────────────


def _render_simulator(predictor) -> None:
    st.markdown("### 🔮 What-If Simulator")
    st.caption(
        "Adjust the inputs and see live predictions from all three models. "
        "`selfMade` is inferred by the classifier — you never need to supply it."
    )

    col_form, col_gap, col_results = st.columns([1.1, 0.1, 1.8])

    # ── Input form ────────────────────────────────────────────────────────
    with col_form:
        st.markdown("#### Inputs")

        final_worth = st.slider(
            "Net worth (billion USD)",
            min_value=1.0,
            max_value=200.0,
            value=5.0,
            step=0.5,
            help="finalWorth in the dataset is in billion USD.",
        )
        age = st.slider(
            "Age",
            min_value=25,
            max_value=100,
            value=52,
        )
        category = st.selectbox(
            "Wealth category",
            options=_CATEGORIES,
            index=0,
        )
        country = st.selectbox(
            "Country",
            options=_COUNTRIES,
            index=0,
        )
        gender = st.radio(
            "Gender",
            options=["M", "F"],
            horizontal=True,
        )

    # ── Results ───────────────────────────────────────────────────────────
    with col_results:
        st.markdown("#### Predictions")

        try:
            result = predictor.predict_all(
                finalWorth=float(final_worth),
                age=float(age),
                category=category,
                country=country,
                gender=gender,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        sm = result["self_made"]
        worth = result["worth"]
        cluster = result["cluster"]

        # ── Self-Made probability gauge ───────────────────────────────────
        st.plotly_chart(
            _probability_gauge(sm["probability"], "Self-Made Probability"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # ── Metric cards ──────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        label_colour = "🟢" if sm["prediction"] == 1 else "🟠"
        m1.metric(
            "Prediction",
            f"{label_colour} {sm['label']}",
        )
        m2.metric(
            "Predicted worth",
            f"${worth['worth_billion_usd']:,.1f}B",
            help="Regressor back-transformed from log scale. "
            "R²=0.08 — treat as indicative, not precise.",
        )
        cluster_id = cluster["cluster"]
        m3.metric(
            "Wealth segment",
            f"Cluster {cluster_id}",
            help=_CLUSTER_LABELS.get(cluster_id, ""),
        )

        # ── Cluster label ─────────────────────────────────────────────────
        st.info(
            f"**Cluster {cluster_id} — "
            f"{_CLUSTER_LABELS.get(cluster_id, 'Wealth Segment')}**  \n"
            f"Silhouette score: {cluster['silhouette']:.3f}  ·  "
            f"k = {cluster['n_clusters']} clusters total"
        )

        # ── Detail expander ───────────────────────────────────────────────
        with st.expander("Raw prediction detail"):
            st.json(result)


# ── Tab 2: Dataset Explorer ────────────────────────────────────────────────


def _render_explorer(df: pd.DataFrame) -> None:
    st.markdown("### 📊 Dataset Explorer")

    # ── KPI row ───────────────────────────────────────────────────────────
    gini_val = _gini(df["finalWorth"].values)
    top1_pct = (
        df.nlargest(int(len(df) * 0.01) or 1, "finalWorth")["finalWorth"].sum()
        / df["finalWorth"].sum()
        * 100
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Billionaires", f"{len(df):,}")
    k2.metric("Median net worth", f"${df['finalWorth'].median():.1f}B")
    k3.metric("Self-made", f"{df['selfMade'].mean()*100:.1f}%")
    k4.metric(
        "Gini coefficient",
        f"{gini_val:.3f}",
        help="1.0 = perfect inequality within the billionaire class.",
    )
    k5.metric("Top 1% wealth share", f"{top1_pct:.1f}%")

    st.divider()

    # ── Row 1: wealth distribution + industry breakdown ───────────────────
    r1c1, r1c2 = st.columns([1.4, 1])

    with r1c1:
        st.markdown("##### Wealth Distribution (log scale)")
        fig_hist = px.histogram(
            df,
            x="log_worth",
            color="selfMade",
            color_discrete_map={1: _PALETTE[0], 0: _PALETTE[1]},
            labels={"log_worth": "log₁₊(Net Worth, $B)", "selfMade": "Self-Made"},
            nbins=50,
            barmode="overlay",
            opacity=0.75,
        )
        fig_hist.update_layout(
            legend_title_text="Self-Made",
            margin=dict(t=10, b=30),
            height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with r1c2:
        st.markdown("##### Billionaires by Industry")
        cat_counts = df["category"].value_counts().head(12).reset_index()
        cat_counts.columns = ["category", "count"]
        fig_bar = px.bar(
            cat_counts,
            x="count",
            y="category",
            orientation="h",
            color="count",
            color_continuous_scale="Blues",
            labels={"count": "Count", "category": ""},
        )
        fig_bar.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
            height=320,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Row 2: geography ──────────────────────────────────────────────────
    st.markdown("##### Geographic Distribution")
    geo_col = st.radio(
        "Colour by",
        ["Billionaire count", "Total wealth ($B)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    country_agg = (
        df.groupby("country")
        .agg(count=("finalWorth", "size"), total_worth=("finalWorth", "sum"))
        .reset_index()
    )
    z_col = "count" if "count" in geo_col else "total_worth"
    z_label = "Count" if "count" in geo_col else "Total Wealth ($B)"

    fig_map = px.choropleth(
        country_agg,
        locations="country",
        locationmode="country names",
        color=z_col,
        color_continuous_scale="Viridis",
        labels={z_col: z_label},
        hover_name="country",
        hover_data={"count": True, "total_worth": ":.1f"},
    )
    fig_map.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        height=380,
        geo=dict(showframe=False, showcoastlines=True),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Row 3: age distribution + Lorenz curve ────────────────────────────
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("##### Age Distribution by Self-Made Status")
        fig_box = px.violin(
            df,
            y="age",
            x="selfMade",
            color="selfMade",
            color_discrete_map={1: _PALETTE[0], 0: _PALETTE[1]},
            box=True,
            points=False,
            labels={"selfMade": "Self-Made", "age": "Age"},
            category_orders={"selfMade": [1, 0]},
        )
        fig_box.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10),
            height=320,
            xaxis=dict(tickvals=[0, 1], ticktext=["Inherited", "Self-Made"]),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with r3c2:
        st.markdown("##### Lorenz Curve — Wealth Inequality")
        x_lorenz, y_lorenz = _lorenz(df["finalWorth"].values)
        fig_lorenz = go.Figure()
        fig_lorenz.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="grey", width=1),
                name="Perfect equality",
            )
        )
        fig_lorenz.add_trace(
            go.Scatter(
                x=x_lorenz,
                y=y_lorenz,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(33, 150, 243, 0.15)",
                line=dict(color=_PALETTE[0], width=2),
                name=f"Billionaires (Gini = {gini_val:.3f})",
            )
        )
        fig_lorenz.update_layout(
            xaxis_title="Cumulative share of billionaires",
            yaxis_title="Cumulative share of wealth",
            legend=dict(x=0.02, y=0.98),
            margin=dict(t=10, b=30),
            height=320,
        )
        st.plotly_chart(fig_lorenz, use_container_width=True)

    # ── Row 4: gender breakdown + top 10 ─────────────────────────────────
    r4c1, r4c2 = st.columns([0.7, 1.3])

    with r4c1:
        st.markdown("##### Gender Split")
        gender_sm = df.groupby(["gender", "selfMade"]).size().reset_index(name="count")
        gender_sm["label"] = gender_sm["selfMade"].map({1: "Self-Made", 0: "Inherited"})
        fig_pie = px.sunburst(
            gender_sm,
            path=["gender", "label"],
            values="count",
            color="label",
            color_discrete_map={
                "Self-Made": _PALETTE[0],
                "Inherited": _PALETTE[1],
            },
        )
        fig_pie.update_layout(
            margin=dict(t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with r4c2:
        st.markdown("##### Top 10 Billionaires by Net Worth")
        top10 = df.nlargest(10, "finalWorth")[
            ["personName", "finalWorth", "category", "country", "selfMade", "age"]
        ].copy()
        top10["selfMade"] = top10["selfMade"].map({1: "✅ Yes", 0: "❌ No"})
        top10["finalWorth"] = top10["finalWorth"].apply(lambda x: f"${x:.1f}B")
        top10 = top10.rename(
            columns={
                "personName": "Name",
                "finalWorth": "Net Worth",
                "category": "Category",
                "country": "Country",
                "selfMade": "Self-Made",
                "age": "Age",
            }
        )
        st.dataframe(top10.reset_index(drop=True), use_container_width=True, height=320)

    # ── Statistical tests summary ─────────────────────────────────────────
    with st.expander("📐 Statistical test results"):
        st.markdown("""
| Test | Question | Result |
|---|---|---|
| Welch's t-test | Self-made vs inherited net worth | Not significant — p = 0.22 |
| One-way ANOVA | Mean worth across top 8 categories | **Significant — F = 4.46, p < 0.001** |
| Chi-squared | Gender × self-made association | **Significant — χ² = 287.2, p < 0.001** |
        """)


# ── Main layout ────────────────────────────────────────────────────────────


def main() -> None:
    # Header
    st.markdown(
        "<h1 style='margin-bottom:0'>💰 Billionaires Statistics</h1>"
        "<p style='color:grey;margin-top:4px'>Advanced Analysis Pipeline — "
        "XGBoost · Optuna · SHAP · K-Means</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Load resources
    data_ok = _DATA_PATH.exists()
    models_ok = all(
        (_PROCESSED / f).exists()
        for f in [
            "classifier.joblib",
            "regressor.joblib",
            "clusterer.joblib",
            "feature_encoder.joblib",
        ]
    )

    if not data_ok:
        st.error(
            f"Dataset not found at `{_DATA_PATH}`.  \n"
            "Download from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/"
            "billionaires-statistics-dataset) and place the CSV at that path."
        )
        st.stop()

    if not models_ok:
        st.warning(
            "Serialised model artifacts not found in `data/processed/`.  \n"
            "Run `python pipeline.py --cluster-k 4` to generate them.  \n"
            "The **Dataset Explorer** tab is still available."
        )

    df = _load_data()

    # Tabs
    tab_sim, tab_exp = st.tabs(["🔮 What-If Simulator", "📊 Dataset Explorer"])

    with tab_sim:
        if not models_ok:
            st.info(
                "Model artifacts are required for the simulator.  \n"
                "Run `python pipeline.py --cluster-k 4` first."
            )
        else:
            predictor = _load_predictor()
            _render_simulator(predictor)

    with tab_exp:
        _render_explorer(df)


if __name__ == "__main__":
    main()
