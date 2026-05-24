# 💰 Billionaires Statistics — Advanced ML Pipeline

[![CI](https://github.com/PRANAVGAWALE-DS/Billionaires-Statistics-Dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/PRANAVGAWALE-DS/Billionaires-Statistics-Dataset/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-blueviolet.svg)](https://optuna.org)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-red.svg)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://huggingface.co/spaces/PG-AIML/billionaires-analysis)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Spaces-live%20demo-yellow.svg)](https://huggingface.co/spaces/PG-AIML/billionaires-analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade ML engineering project built on the [Kaggle Billionaires Statistics Dataset (2023)](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset) — from raw CSV to a live interactive demo. Covers the full ML lifecycle: rigorous EDA, statistical testing, three XGBoost models with Optuna HPO and SHAP explainability, a reproducible 7-step training pipeline, 162-test CI suite, FastAPI serving layer, and a Streamlit app deployed on Hugging Face Spaces.

---

## 🚀 Live Demo

**[→ Try the Streamlit app on Hugging Face Spaces](https://huggingface.co/spaces/PG-AIML/billionaires-analysis)**

| Tab | What you can do |
|---|---|
| 🔮 **What-If Simulator** | Adjust age, net worth, country, category — get live predictions from all three models instantly |
| 📊 **Dataset Explorer** | Interactive choropleth map, Lorenz inequality curve, wealth distributions, gender sunburst, top-10 table |

---

## Results at a Glance

| Model | Metric | Val | Test |
|---|---|---|---|
| Self-Made Classifier | ROC-AUC | 0.8505 | **0.8043** |
| Self-Made Classifier | F1 (Self-Made class)¹ | 0.8510 | **0.8656** |
| Self-Made Classifier | Accuracy | 0.7727 | **0.7929** |
| Net-Worth Regressor | R² | 0.0642 | **0.0487** ² |
| Net-Worth Regressor | MAE (log scale) | 0.5864 | **0.6449** |
| Wealth Clusterer | Silhouette (k=4) | — | **0.2305** |
| Wealth Clusterer | Davies-Bouldin (k=4) | — | **1.5941** |

> ¹ F1 reported for the **Self-Made (positive) class** only. Macro-F1 is 0.72. Inherited recall is lower (≈ 0.29 on test) due to class imbalance — the HPO search space includes `scale_pos_weight` to balance this trade-off.
>
> ² Predicting net-worth magnitude from age, gender, and geography is a genuinely hard problem — these features carry limited signal about wealth scale. The low R² is honest, not a failure. SHAP plots reveal which features drive the predictions that are made. The regressor is presented in the app with an explicit uncertainty disclaimer.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Pipeline](#pipeline)
6. [Serving Layer](#serving-layer)
7. [Notebooks](#notebooks)
8. [ML Models](#ml-models)
9. [Test Suite](#test-suite)
10. [Analysis Highlights](#analysis-highlights)
11. [Key Design Decisions](#key-design-decisions)
12. [Hardware Notes](#hardware-notes)

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — nelgiriyewithana](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset) |
| Records | 2,640 billionaires |
| Raw columns | 35 |
| Engineered columns | 43 (after `build_features`) |
| Key columns | `finalWorth`, `age`, `selfMade`, `category`, `country`, `gender` |

> **Unit convention:** The raw CSV stores `finalWorth` in **millions USD**. The API and Streamlit app accept inputs in **billions USD**. `predictor._build_row()` multiplies by 1,000 before feature construction so all model inputs stay on the training scale.

### Data quality summary

| Column | Issue | Handling |
|---|---|---|
| `age` | 2.46% missing | Per-category median imputation |
| `country` | 1.44% missing | Retained as-is; unknown countries map to continent "Other" |
| `organization` / `title` | ~87% missing | Excluded from modelling |
| `selfMade` | Raw dtype `object` | Cast to `int64` in `loader.clean()` |

---

## Architecture

```
CSV  (finalWorth in millions USD)
 │
 ▼
loader.py          load_raw → validate_schema → clean
 │
 ▼
engineer.py        build_features(encode=False)
 │                 ├── log_worth, age_group, wealth_per_decade, continent
 │                 └── [no encoding yet — split-safe]
 │
 ▼
pipeline.py        stratified 70/15/15 split (on selfMade)
 │
 ▼
engineer.py        FeatureEncoder.fit(train) → transform(train, val, test, full)
 │                 OrdinalEncoder — unknown categories → -1, no crash
 │
 ├──▶ classifier.py    SelfMadeClassifier  tune → fit → evaluate
 ├──▶ regressor.py     WorthRegressor      tune → fit → evaluate
 └──▶ clusterer.py     BillionaireClusterer fit(full dataset) → evaluate
           │
           ▼
      evaluate.py    evaluate_classifier / evaluate_regressor / evaluate_clusters
           │
           ▼
      data/processed/   9 serialised artifacts (.joblib + XGBoost native .json)
           │
           ├──▶ api/main.py       FastAPI — 6 endpoints, Swagger UI
           └──▶ app.py            Streamlit — What-If Simulator + Dataset Explorer
```

---

## Project Structure

```
billionaires-analysis/
├── data/
│   ├── raw/
│   │   └── Billionaires Statistics Dataset.csv   ← place dataset here
│   └── processed/                                ← auto-populated by pipeline.py
│       ├── classifier.joblib
│       ├── classifier_xgb.json
│       ├── regressor.joblib
│       ├── regressor_xgb.json
│       ├── clusterer.joblib
│       ├── feature_encoder.joblib
│       ├── feature_cols.json
│       └── metrics.json
├── notebooks/
│   ├── 01_eda.ipynb          ← EDA, stats, inequality, maps
│   └── 02_modeling.ipynb     ← ML, SHAP, clustering
├── reports/
│   └── figures/              ← auto-populated on notebook run
├── src/
│   └── billionaires/
│       ├── data/
│       │   └── loader.py     ← load_raw, validate_schema, clean
│       ├── features/
│       │   └── engineer.py   ← build_features, FeatureEncoder, feature lists
│       ├── models/
│       │   ├── classifier.py ← SelfMadeClassifier (XGBoost + Optuna)
│       │   ├── regressor.py  ← WorthRegressor     (XGBoost + Optuna)
│       │   ├── clusterer.py  ← BillionaireClusterer (K-Means + PCA)
│       │   └── evaluate.py   ← evaluate_classifier/regressor/clusters
│       └── viz/
│           └── plots.py      ← reusable figure functions (shap imported lazily)
├── api/
│   ├── main.py               ← FastAPI app (6 endpoints, lifespan loading)
│   ├── predictor.py          ← BillionairesPredictor (artifact loading + inference)
│   └── schemas.py            ← Pydantic request / response models
├── tests/
│   ├── conftest.py           ← shared fixtures (synthetic 120-row DataFrame)
│   ├── test_loader.py        ← 17 tests
│   ├── test_engineer.py      ← 27 tests
│   ├── test_classifier.py    ← 19 tests
│   ├── test_regressor.py     ← 20 tests
│   ├── test_clusterer.py     ← 26 tests
│   ├── test_predictor.py     ← 35 tests  (incl. unit-scale regression guard)
│   └── test_api.py           ← 24 tests  (FastAPI TestClient, schema validation)
├── .github/
│   └── workflows/
│       └── ci.yml            ← lint → test on Python 3.10, 3.11, 3.12
├── app.py                    ← Streamlit app (What-If + Dataset Explorer)
├── pipeline.py               ← end-to-end training + serialisation CLI
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1 — Clone
git clone https://github.com/PRANAVGAWALE-DS/Billionaires-Statistics-Dataset.git
cd billionaires-analysis

# 2 — Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3 — Install package in editable mode with dev + api dependencies
pip install -e ".[dev,api]"

# 4 — Place the dataset
# Download from Kaggle and save at:
# data/raw/Billionaires Statistics Dataset.csv

# 5 — Run the full training pipeline (~66s on CPU)
python pipeline.py --cluster-k 4

# 6 — Launch the Streamlit app
streamlit run app.py

# 7 — Or start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 8 — Run the full test suite (162 tests)
pytest
```

---

## Pipeline

`pipeline.py` is the single entry point for reproducible end-to-end training. It orchestrates all 7 steps, serialises 9 artifacts to `data/processed/`, and prints a formatted results summary.

```
Step 1/7  Load + validate + clean
Step 2/7  Build base features  (encode=False — split-safe)
Step 3/7  Stratified 70/15/15 split on selfMade
Step 4/7  Fit FeatureEncoder on training split only
Step 5/7  SelfMadeClassifier — tune → fit → evaluate
Step 6/7  WorthRegressor     — tune → fit → evaluate
Step 7/7  BillionaireClusterer — fit(full dataset) → evaluate → serialise all
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--data-path` | `data/raw/Billionaires Statistics Dataset.csv` | Path to raw CSV |
| `--out-dir` | `data/processed` | Artifact output directory |
| `--n-trials` | `40` | Optuna HPO trials per model |
| `--cv-folds` | `5` | Cross-validation folds during HPO |
| `--seed` | `42` | Global random seed (`random` + `numpy` + all model states) |
| `--cluster-k` | auto | Fix K-Means k (skips elbow search) |
| `--skip-hpo` | off | Skip Optuna; use XGBoost defaults (fast dev mode, ~30s) |

```bash
# Full production run (~66s on CPU)
python pipeline.py --cluster-k 4

# Fast dev / CI iteration (~30s)
python pipeline.py --skip-hpo --cluster-k 4
```

### Serialised artifacts

| File | Format | Use |
|---|---|---|
| `classifier.joblib` | joblib | Full Python wrapper — `predict_proba` |
| `classifier_xgb.json` | XGBoost native | Portable, ONNX-ready |
| `regressor.joblib` | joblib | Full wrapper — `predict(exponentiate=True)` |
| `regressor_xgb.json` | XGBoost native | Portable, ONNX-ready |
| `clusterer.joblib` | joblib | Full wrapper — scaler + PCA + KMeans |
| `feature_encoder.joblib` | joblib | Fitted `OrdinalEncoder` |
| `feature_cols.json` | JSON | **Authoritative** feature column manifest per model |
| `metrics.json` | JSON | All evaluation metrics (read by app tooltip at runtime) |

---

## Serving Layer

### FastAPI — REST API

Start the server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI → http://localhost:8000/docs
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model load status — 503 if artifacts missing |
| `GET` | `/metrics` | Stored evaluation metrics from `metrics.json` |
| `POST` | `/predict` | All three models in one call (**preferred**) |
| `POST` | `/predict/self-made` | Classifier only — P(selfMade) |
| `POST` | `/predict/worth` | Regressor only — log + dollar-scale |
| `POST` | `/predict/cluster` | Clusterer only — wealth segment |

> **Prefer `POST /predict`** when you need results from more than one model. The individual endpoints each run the classifier independently to infer `selfMade` — calling them separately runs the classifier twice.

Example request (`finalWorth` in **billion USD**):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"finalWorth":5.2,"age":52,"category":"Technology","country":"United States","gender":"M"}'
```

```json
{
  "self_made":  { "probability": 0.9794, "prediction": 1,  "label": "Self-Made" },
  "worth":      { "log_worth_predicted": 7.9373, "worth_billion_usd": 2.7987, "selfMade_used": 1 },
  "cluster":    { "cluster": 2, "n_clusters": 4, "silhouette": 0.2305 }
}
```

> `selfMade` is never provided by the caller — it is inferred by the classifier and propagated as a feature to the regressor and clusterer. `WorthResponse.selfMade_used` makes this transparent.
>
> `WorthResponse` carries an explicit accuracy caveat: R²≈0.05 on the test set. `worth_billion_usd` reflects population-level trends and should be treated as a directional estimate, not a point forecast.

### Streamlit App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

| Tab | Content |
|---|---|
| **What-If Simulator** | Net worth slider (billion USD), age slider, category / country dropdowns, gender toggle → probability gauge, metric cards, cluster label, raw JSON expander |
| **Dataset Explorer** | 5 KPI cards · wealth histogram · ISO-3 choropleth map · industry bar chart · age violin · Lorenz curve · gender sunburst · top-10 table · statistical tests |

**[→ Live on Hugging Face Spaces](https://huggingface.co/spaces/PG-AIML/billionaires-analysis)**

---

## Notebooks

### `01_eda.ipynb` — Exploratory Data Analysis

| Section | Detail |
|---|---|
| Data quality | Missing-value audit, schema validation, dtype inspection |
| Cleaning | Per-category median age imputation, type coercion, deduplication |
| Feature engineering | `log_worth`, `age_group`, `wealth_per_decade`, `continent`, label-encoded categoricals |
| Distributions | Raw vs log(1+x) wealth · age by self-made status (violin + box) |
| Statistical tests | t-test (self-made vs inherited) · ANOVA (by category) · Chi-squared (gender × self-made) |
| Wealth inequality | Gini coefficient, Lorenz curve, top-1% / top-10% concentration |
| Geospatial | Plotly choropleth — billionaire count and total wealth by country (ISO-3) |

### `02_modeling.ipynb` — Machine Learning

| Section | Detail |
|---|---|
| Classification | Predict `selfMade` — XGBoost + Optuna HPO (incl. `scale_pos_weight`) |
| Regression | Predict `log_worth` — XGBoost + Optuna HPO |
| Evaluation | `evaluate_classifier` / `evaluate_regressor` from `evaluate.py` |
| Explainability | SHAP `TreeExplainer` — bar and beeswarm for both models |
| Clustering | `BillionaireClusterer` — silhouette-selected k, PCA 2D scatter, cluster profiles |

---

## ML Models

### 1 · Self-Made Classifier

Predicts whether a billionaire is self-made (`1`) or inherited (`0`).

| Property | Detail |
|---|---|
| Algorithm | XGBoost binary classifier |
| HPO | Optuna — 40 trials, 5-fold stratified CV, optimises ROC-AUC |
| HPO search space | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, **`scale_pos_weight`** |
| Split | 70% train / 15% val (early stopping) / 15% test |
| Features | `age`, `log_worth`, `wealth_per_decade`, `category_enc`, `country_enc`, `gender_enc`, `continent_enc` |
| **ROC-AUC (test)** | **0.8043** |
| **F1 — Self-Made class (test)** | **0.8656** |
| **Accuracy (test)** | **0.7929** |
| Class imbalance | Self-Made : Inherited ≈ 2.2 : 1 — `scale_pos_weight` tuned by Optuna |
| Explainability | SHAP `TreeExplainer` — beeswarm + bar |

### 2 · Net-Worth Regressor

Predicts `log(1 + finalWorth_millions)` from demographic and geographic features.

| Property | Detail |
|---|---|
| Algorithm | XGBoost regression |
| HPO | Optuna — 40 trials, 5-fold KFold CV, optimises R² |
| Split | 70% train / 15% val (early stopping) / 15% test |
| Features | `age`, `selfMade`, `category_enc`, `country_enc`, `gender_enc`, `continent_enc` |
| **R² (test)** | **0.0487** |
| **MAE (test, log scale)** | **0.6449** |
| **MAPE (test, dollar scale)** | **59.8%** |
| Leakage guard | `wealth_per_decade` excluded — its numerator *is* the target |
| Uncertainty | R²≈0.05 — surfaced explicitly in API response schema and app tooltip |
| Explainability | SHAP `TreeExplainer` — dot plot |

### 3 · Wealth Segment Clusterer

Groups billionaires into wealth segments using K-Means.

| Property | Detail |
|---|---|
| Algorithm | K-Means + StandardScaler (distance-based — scaling mandatory) |
| K selection | Max silhouette score across k=2..10, or fixed via `--cluster-k` |
| `n_init` | 10 (stable centroid initialisation — OpenMP guard prevents Windows deadlock) |
| Training scope | Full dataset (train + val + test) — intentional for exploratory segmentation |
| Visualisation | PCA 2D projection (41.2% explained variance at k=4) |
| **Silhouette (k=4)** | **0.2305** |
| **Davies-Bouldin (k=4)** | **1.5941** |
| Profiles | Per-cluster mean / median for all features |

> The clusterer is fitted on the full dataset (not just the training split) because K-Means here serves exploratory wealth segmentation, not predictive modelling. Maximising the number of billionaires visible to the centroid algorithm produces more stable and interpretable segments. This is noted in `pipeline.py`, `predictor.py`, and `schemas.py`.

---

## Test Suite

162 tests across 7 modules, organised by class. Run with:

```bash
pytest                           # all 162 tests with coverage report
pytest tests/test_loader.py -v   # single module
pytest tests/test_api.py -v      # API route tests
```

| File | Tests | Covers |
|---|---|---|
| `test_loader.py` | 17 | `load_raw`, `validate_schema`, `clean` |
| `test_engineer.py` | 27 | `build_features`, `FeatureEncoder`, feature list contracts |
| `test_classifier.py` | 19 | Instantiation, fit, predict, proba, evaluate |
| `test_regressor.py` | 20 | Instantiation, fit, predict, exponentiate, evaluate |
| `test_clusterer.py` | 26 | fit, elbow search, PCA, predict, profiles, `n_init` regression guard |
| `test_predictor.py` | 35 | Artifact loading, **unit-scale regression guard (C1)**, feature column alignment, all `predict_*` contracts |
| `test_api.py` | 24 | All 6 FastAPI routes, schema rejection (422), 503 on missing models |

All tests use a 120-row synthetic `DataFrame` defined in `conftest.py` — no CSV or pre-trained artifacts required. CI runs on every push and PR via GitHub Actions (`ruff check` → `ruff format --check` → `pytest --cov --cov-fail-under=70`) across **Python 3.10, 3.11, and 3.12**.

### Key regression guard

`test_predictor.py::TestBuildRowUnitConversion` asserts that `log_worth` computed inside `_build_row()` falls within the training distribution (`[6.0, 14.0]`). If the billion→million unit conversion is removed, `log1p(5.2)` ≈ 1.8 — completely outside the training range — and this test fails immediately.

---

## Analysis Highlights

### Statistical tests

| Test | Question | Result |
|---|---|---|
| Welch's t-test | Do self-made and inherited billionaires differ in net worth? | Not significant — p = 0.22 |
| One-way ANOVA | Do mean net worths differ across the top 8 categories? | **Significant — F = 4.46, p < 0.001** |
| Chi-squared | Is gender associated with self-made status? | **Significant — χ² = 287.2, p < 0.001** |

### Key findings

| Metric | Value |
|---|---|
| Dataset size (after cleaning) | 2,640 billionaires |
| Self-made share | 68.6% |
| Median net worth | $2.3B |
| Top 1% wealth share | 17.6% |
| Gini coefficient | 0.549 (extreme concentration within the billionaire class) |
| Gender split | ~85% male · ~15% female |
| Female self-made rate | 28.5% vs male 72.5% — χ² significant |

---

## Key Design Decisions

**Thin notebooks, fat `src/`** — all reusable logic lives in the `billionaires` package. Notebooks import and call; they never define. This makes every function unit-testable and importable without copy-pasting.

**Split-aware encoding via `FeatureEncoder`** — `build_features(encode=False)` is called before the train/test split. A dedicated `FeatureEncoder` (wrapping `OrdinalEncoder` with `handle_unknown="use_encoded_value"`) is fitted only on the training split. Unknown categories at inference time return `-1` without raising errors.

**`feature_cols.json` as the serving contract** — `BillionairesPredictor.load()` reads column lists from the serialised JSON manifest, not from the installed package's module functions. This ensures the serving layer always uses the exact feature set frozen at training time, regardless of any subsequent changes to `engineer.py`.

**Unit convention: billions in, millions trained** — The API and Streamlit app accept `finalWorth` in billion USD (intuitive for users). `_build_row()` multiplies by 1,000 before calling `build_features()` so all features stay on the millions scale the model was trained on. A regression test in `test_predictor.py` guards this conversion permanently.

**Log-transform the target** — `finalWorth` has skewness > 10. The regressor trains on `log1p(finalWorth_millions)`. `WorthRegressor.predict(exponentiate=True)` converts back to the original dollar scale. The API divides by 1,000 to return billions.

**`scale_pos_weight` in HPO** — The Self-Made : Inherited class ratio is ~2.2:1. Including `scale_pos_weight` in the Optuna search space allows the trial sampler to find the class-weighting that maximises ROC-AUC, rather than defaulting to 1.0 (which suppresses Inherited recall).

**Per-group age imputation** — age is imputed with the within-`category` median rather than the global mean. A 40-year-old tech founder and an 80-year-old manufacturing heir belong to different age distributions.

**Leakage guard on `wealth_per_decade`** — this feature (`finalWorth / (age / 10)`) is a valid predictor for the classifier (net worth is observable before asking *how* someone became wealthy) but is excluded from the regressor's feature list since its numerator is the prediction target.

**Early stopping on a dedicated val set** — both models use a 70/15/15 split. The val set drives `early_stopping_rounds=20`; the test set is never seen during training or stopping decisions.

**`n_init=10` for stable K-Means** — `OMP_NUM_THREADS=1` is set in `pipeline.py`, `app.py`, and `ci.yml`, which eliminates the Windows OpenMP thread-pool deadlock. With that guard in place, `n_init=10` is safe and produces materially more stable centroids than `n_init=1`.

**FastAPI lifespan over `@app.on_event`** — the modern FastAPI pattern loads all three joblib objects once at startup into `app.state.predictor`. No per-request loading, no global variables. `/health` returns 503 — not 200 — when models are not loaded, which matters for load balancers and readiness probes.

**`selfMade` inferred, never requested** — the API caller only provides observable attributes. `selfMade` is predicted by the classifier and propagated as a feature to the regressor and clusterer. `WorthResponse.selfMade_used` exposes this transparently.

**Lazy `shap` import** — `import shap` is deferred inside `shap_summary()` rather than placed at module level. This avoids loading a ~100 MB optional dependency in lightweight inference containers or CI environments that never call the SHAP plot.

---

## Hardware Notes

- Developed on Windows 11, NVIDIA RTX 3050 4GB VRAM
- XGBoost runs on CPU by default — add `device="cuda"` at model construction for GPU acceleration
- Full pipeline (`--cluster-k 4`, 40 HPO trials per model) completes in **~66 seconds on CPU**
- Fast dev mode (`--skip-hpo --cluster-k 4`) completes in **~30 seconds**
- `OMP_NUM_THREADS=1` is set in `pipeline.py`, `conftest.py`, `app.py`, and `ci.yml` to prevent OpenMP thread-pool deadlocks on Windows

---

## License

MIT — see [LICENSE](LICENSE).