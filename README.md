# 💰 Billionaires Statistics — Advanced ML Pipeline

[![CI](https://github.com/PRANAVGAWALE-DS/Billionaires-Statistics-Dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/PRANAVGAWALE-DS/Billionaires-Statistics-Dataset/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-blueviolet.svg)](https://optuna.org)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-red.svg)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://huggingface.co/spaces/PG-AIML/billionaires-analysis)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Spaces-live%20demo-yellow.svg)](https://huggingface.co/spaces/PG-AIML/billionaires-analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade ML engineering project built on the [Kaggle Billionaires Statistics Dataset (2023)](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset) — from raw CSV to a live interactive demo. Covers the full ML lifecycle: rigorous EDA, statistical testing, three XGBoost models with Optuna HPO and SHAP explainability, a reproducible 7-step training pipeline, 83-test CI suite, FastAPI serving layer, and a Streamlit app deployed on Hugging Face Spaces.

---

## 🚀 Live Demo

**[→ Try the Streamlit app on Hugging Face Spaces](https://huggingface.co/spaces/PG-AIML/billionaires-analysis)**

| Tab | What you can do |
|---|---|
| 🔮 **What-If Simulator** | Adjust age, net worth, country, category — get live predictions from all three models instantly |
| 📊 **Dataset Explorer** | Interactive choropleth map, Lorenz inequality curve, wealth distributions, gender sunburst, top-10 table |

---

## Results at a glance

| Model | Metric | Value |
|---|---|---|
| Self-Made Classifier | ROC-AUC (test) | **0.8154** |
| Self-Made Classifier | F1 (test) | **0.8454** |
| Self-Made Classifier | Accuracy (test) | **0.7727** |
| Net-Worth Regressor | R² (test) | 0.0487 ¹ |
| Net-Worth Regressor | MAE (log scale) | 0.6449 |
| Wealth Clusterer | Silhouette (k=4) | 0.1790 |

> ¹ Predicting net-worth magnitude from age, gender, and geography is a genuinely hard problem — these features carry limited signal about wealth scale. The low R² is honest, not a failure. SHAP plots reveal which features drive the predictions that are made.

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

### Data quality summary

| Column | Issue | Handling |
|---|---|---|
| `age` | 2.46% missing | Per-category median imputation |
| `country` | 1.44% missing | Retained as-is |
| `organization` / `title` | ~87% missing | Excluded from modelling |
| `selfMade` | Raw dtype `object` | Cast to `int64` in `loader.clean()` |

---

## Architecture

```
CSV
 |
 v
loader.py          load_raw -> validate_schema -> clean
 |
 v
engineer.py        build_features(encode=False)
 |                 +-- log_worth, age_group, wealth_per_decade, continent
 |                 L-- [no encoding yet -- split-safe]
 |
 v
pipeline.py        stratified 70/15/15 split (on selfMade)
 |
 v
engineer.py        FeatureEncoder.fit(train) -> transform(train, val, test, full)
 |                 OrdinalEncoder -- unknown categories -> -1, no crash
 |
 +---> classifier.py    SelfMadeClassifier  tune -> fit -> evaluate
 +---> regressor.py     WorthRegressor      tune -> fit -> evaluate
 L---> clusterer.py     BillionaireClusterer fit(full dataset) -> evaluate
          |
          v
     evaluate.py    evaluate_classifier / evaluate_regressor / evaluate_clusters
          |
          v
     data/processed/   8 serialised artifacts  (.joblib + XGBoost native .json)
          |
          +---> api/main.py       FastAPI -- 6 endpoints, Swagger UI
          L---> app.py            Streamlit -- What-If Simulator + Dataset Explorer
```

---

## Project Structure

```
billionaires-analysis/
+-- data/
|   +-- raw/
|   |   L-- Billionaires Statistics Dataset.csv   <- place dataset here
|   L-- processed/                                <- auto-populated by pipeline.py
|       +-- classifier.joblib
|       +-- classifier_xgb.json
|       +-- regressor.joblib
|       +-- regressor_xgb.json
|       +-- clusterer.joblib
|       +-- feature_encoder.joblib
|       +-- feature_cols.json
|       L-- metrics.json
+-- notebooks/
|   +-- 01_eda.ipynb          <- EDA, stats, inequality, maps
|   L-- 02_modeling.ipynb     <- ML, SHAP, clustering
+-- reports/
|   L-- figures/              <- auto-populated on notebook run
+-- src/
|   L-- billionaires/
|       +-- data/
|       |   L-- loader.py     <- load_raw, validate_schema, clean
|       +-- features/
|       |   L-- engineer.py   <- build_features, FeatureEncoder, feature lists
|       +-- models/
|       |   +-- classifier.py <- SelfMadeClassifier (XGBoost + Optuna)
|       |   +-- regressor.py  <- WorthRegressor     (XGBoost + Optuna)
|       |   +-- clusterer.py  <- BillionaireClusterer (K-Means + PCA)
|       |   L-- evaluate.py   <- evaluate_classifier/regressor/clusters
|       L-- viz/
|           L-- plots.py      <- reusable figure functions
+-- api/
|   +-- main.py               <- FastAPI app (6 endpoints, lifespan loading)
|   +-- predictor.py          <- BillionairesPredictor (artifact loading + inference)
|   L-- schemas.py            <- Pydantic request / response models
+-- tests/
|   +-- conftest.py           <- shared fixtures (synthetic 120-row DataFrame)
|   +-- test_loader.py        <- 17 tests
|   +-- test_engineer.py      <- 27 tests
|   +-- test_classifier.py    <- 19 tests
|   L-- test_regressor.py     <- 20 tests
+-- .github/
|   L-- workflows/
|       L-- ci.yml            <- lint -> test on every push / PR
+-- app.py                    <- Streamlit app (What-If + Dataset Explorer)
+-- pipeline.py               <- end-to-end training + serialisation CLI
+-- pyproject.toml
+-- requirements.txt
L-- README.md
```

---

## Quick Start

```bash
# 1 -- Clone
git clone https://github.com/PRANAVGAWALE-DS/Billionaires-Statistics-Dataset.git
cd billionaires-analysis

# 2 -- Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3 -- Install package in editable mode with dev + api dependencies
pip install -e ".[dev,api]"

# 4 -- Place the dataset
# Download from Kaggle and save at:
# data/raw/Billionaires Statistics Dataset.csv

# 5 -- Run the full training pipeline
python pipeline.py --cluster-k 4

# 6 -- Launch the Streamlit app
streamlit run app.py

# 7 -- Or start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 8 -- Run the test suite
pytest
```

---

## Pipeline

`pipeline.py` is the single entry point for reproducible end-to-end training. It orchestrates all 7 steps, serialises 8 artifacts to `data/processed/`, and prints a formatted results summary.

```
Step 1/7  Load + validate + clean
Step 2/7  Build base features  (encode=False -- split-safe)
Step 3/7  Stratified 70/15/15 split on selfMade
Step 4/7  Fit FeatureEncoder on training split only
Step 5/7  SelfMadeClassifier -- tune -> fit -> evaluate
Step 6/7  WorthRegressor     -- tune -> fit -> evaluate
Step 7/7  BillionaireClusterer -- fit -> evaluate -> serialise all
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--data-path` | `data/raw/Billionaires Statistics Dataset.csv` | Path to raw CSV |
| `--out-dir` | `data/processed` | Artifact output directory |
| `--n-trials` | `40` | Optuna HPO trials per model |
| `--cv-folds` | `5` | Cross-validation folds during HPO |
| `--seed` | `42` | Global random seed |
| `--cluster-k` | auto | Fix K-Means k (skips elbow search) |
| `--skip-hpo` | off | Skip Optuna; use XGBoost defaults (fast dev mode) |

```bash
# Full production run (~3 min on CPU)
python pipeline.py --cluster-k 4

# Fast dev / CI iteration (~30 sec)
python pipeline.py --skip-hpo --cluster-k 4
```

### Serialised artifacts

| File | Format | Use |
|---|---|---|
| `classifier.joblib` | joblib | Full Python wrapper -- `predict_proba` |
| `classifier_xgb.json` | XGBoost native | Portable, ONNX-ready |
| `regressor.joblib` | joblib | Full wrapper -- `predict(exponentiate=True)` |
| `regressor_xgb.json` | XGBoost native | Portable, ONNX-ready |
| `clusterer.joblib` | joblib | Full wrapper -- scaler + PCA + KMeans |
| `feature_encoder.joblib` | joblib | Fitted `OrdinalEncoder` |
| `feature_cols.json` | JSON | Feature column manifest per model |
| `metrics.json` | JSON | All evaluation metrics |

---

## Serving Layer

### FastAPI -- REST API

Start the server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI -> http://localhost:8000/docs
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model load status -- 503 if artifacts missing |
| `GET` | `/metrics` | Stored evaluation metrics from `metrics.json` |
| `POST` | `/predict` | All three models in one call |
| `POST` | `/predict/self-made` | Classifier only -- P(selfMade) |
| `POST` | `/predict/worth` | Regressor only -- log + dollar-scale |
| `POST` | `/predict/cluster` | Clusterer only -- wealth segment |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"finalWorth":5.2,"age":52,"category":"Technology","country":"United States","gender":"M"}'
```

```json
{
  "self_made":  { "probability": 0.9794, "prediction": 1, "label": "Self-Made" },
  "worth":      { "log_worth_predicted": 7.9373, "worth_billion_usd": 2798.71, "selfMade_used": 1 },
  "cluster":    { "cluster": 3, "n_clusters": 4, "silhouette": 0.179 }
}
```

> `selfMade` is never provided by the caller -- it is inferred by the classifier and propagated as a feature to the regressor and clusterer. `WorthResponse.selfMade_used` makes this transparent.

### Streamlit App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

| Tab | Content |
|---|---|
| **What-If Simulator** | Net worth slider, age slider, category / country dropdowns, gender toggle -> probability gauge, metric cards, cluster label, raw JSON expander |
| **Dataset Explorer** | 5 KPI cards · wealth histogram · choropleth map · industry bar chart · age violin · Lorenz curve · gender sunburst · top-10 table · statistical tests |

**[-> Live on Hugging Face Spaces](https://huggingface.co/spaces/PG-AIML/billionaires-analysis)**

---

## Notebooks

### `01_eda.ipynb` -- Exploratory Data Analysis

| Section | Detail |
|---|---|
| Data quality | Missing-value audit, schema validation, dtype inspection |
| Cleaning | Per-category median age imputation, type coercion, deduplication |
| Feature engineering | `log_worth`, `age_group`, `wealth_per_decade`, `continent`, label-encoded categoricals |
| Distributions | Raw vs log(1+x) wealth · age by self-made status (violin + box) |
| Statistical tests | t-test (self-made vs inherited) · ANOVA (by category) · Chi-squared (gender x self-made) |
| Wealth inequality | Gini coefficient, Lorenz curve, top-1% / top-10% concentration |
| Geospatial | Plotly choropleth -- billionaire count and total wealth by country |

### `02_modeling.ipynb` -- Machine Learning

| Section | Detail |
|---|---|
| Classification | Predict `selfMade` -- XGBoost + Optuna HPO |
| Regression | Predict `log_worth` -- XGBoost + Optuna HPO |
| Evaluation | `evaluate_classifier` / `evaluate_regressor` from `evaluate.py` |
| Explainability | SHAP `TreeExplainer` -- bar and beeswarm for both models |
| Clustering | `BillionaireClusterer` -- silhouette-selected k, PCA 2D scatter, cluster profiles |

---

## ML Models

### 1 · Self-Made Classifier

Predicts whether a billionaire is self-made (`1`) or inherited (`0`).

| Property | Detail |
|---|---|
| Algorithm | XGBoost binary classifier |
| HPO | Optuna -- 40 trials, 5-fold stratified CV, optimises ROC-AUC |
| Split | 70% train / 15% val (early stopping) / 15% test |
| Features | `age`, `log_worth`, `wealth_per_decade`, `category_enc`, `country_enc`, `gender_enc`, `continent_enc` |
| **ROC-AUC (test)** | **0.8154** |
| **F1 (test)** | **0.8454** |
| **Accuracy (test)** | **0.7727** |
| Explainability | SHAP `TreeExplainer` -- beeswarm + bar |

### 2 · Net-Worth Regressor

Predicts `log(1 + finalWorth)` from demographic and geographic features.

| Property | Detail |
|---|---|
| Algorithm | XGBoost regression |
| HPO | Optuna -- 40 trials, 5-fold KFold CV, optimises R² |
| Split | 70% train / 15% val (early stopping) / 15% test |
| Features | `age`, `selfMade`, `category_enc`, `country_enc`, `gender_enc`, `continent_enc` |
| **R² (test)** | **0.0487** |
| **MAE (test, log scale)** | **0.6449** |
| Explainability | SHAP `TreeExplainer` -- dot plot |
| Leakage guard | `wealth_per_decade` excluded -- encodes `finalWorth` directly |

### 3 · Wealth Segment Clusterer

Groups billionaires into wealth segments using K-Means.

| Property | Detail |
|---|---|
| Algorithm | K-Means with StandardScaler pre-processing |
| K selection | Max silhouette score across k=2..10 (or fixed via `--cluster-k`) |
| Visualisation | PCA 2D projection (41% explained variance at k=4) |
| **Silhouette (k=4)** | **0.1790** |
| **Davies-Bouldin (k=4)** | **1.7010** |
| Profiles | Per-cluster mean / median for all features |

---

## Test Suite

83 tests across 4 modules, organised by class. Run with:

```bash
pytest                           # all 83 tests
pytest tests/test_loader.py -v   # single module
```

| File | Tests | Covers |
|---|---|---|
| `test_loader.py` | 17 | `load_raw`, `validate_schema`, `clean` |
| `test_engineer.py` | 27 | `build_features`, `FeatureEncoder`, feature list contracts |
| `test_classifier.py` | 19 | Instantiation, fit, predict, proba, evaluate |
| `test_regressor.py` | 20 | Instantiation, fit, predict, exponentiate, evaluate |

All tests use a 120-row synthetic `DataFrame` defined in `conftest.py` -- no CSV required. CI runs on every push and PR via GitHub Actions (`ruff check` -> `ruff format --check` -> `pytest`).

---

## Analysis Highlights

### Statistical tests

| Test | Question | Result |
|---|---|---|
| Welch's t-test | Do self-made and inherited billionaires differ in net worth? | Not significant -- p = 0.22 |
| One-way ANOVA | Do mean net worths differ across the top 8 categories? | Significant -- F = 4.46, p < 0.001 |
| Chi-squared | Is gender associated with self-made status? | Significant -- chi2 = 287.2, p < 0.001 |

### Key findings

| Metric | Value |
|---|---|
| Dataset size (after cleaning) | 2,640 billionaires |
| Self-made share | 69.2% |
| Median net worth | $2.3B |
| Top 1% wealth share | 18.0% |
| Top 10% wealth share | 47.8% |
| Gini coefficient | > 0.6 (extreme concentration within the billionaire class) |
| Gender split | 85.3% male · 14.7% female |
| Female self-made rate | 28.5% vs male 72.5% -- chi2 significant |

---

## Key Design Decisions

**Thin notebooks, fat `src/`** -- all reusable logic lives in the `billionaires` package. Notebooks import and call; they never define. This makes every function unit-testable and importable without copy-pasting.

**Split-aware encoding via `FeatureEncoder`** -- `build_features(encode=False)` is called before the train/test split. A dedicated `FeatureEncoder` class (wrapping `OrdinalEncoder` with `handle_unknown="use_encoded_value"`) is then fitted only on the training split. This prevents categorical encoding leakage into the held-out set. Unknown categories at inference time return `-1` without raising errors.

**Log-transform the target** -- `finalWorth` has skewness > 10. The regressor trains on `log1p(finalWorth)`. `WorthRegressor.predict(exponentiate=True)` converts back to the original dollar scale.

**Per-group age imputation** -- age is imputed with the within-`category` median rather than the global mean. A 40-year-old tech founder and an 80-year-old manufacturing heir belong to different age distributions.

**Leakage guard on `wealth_per_decade`** -- this feature (`finalWorth / (age / 10)`) is a valid predictor for the classifier but is excluded from the regressor's feature list since its numerator is the prediction target.

**Early stopping on a dedicated val set** -- both models use a 70/15/15 split. The val set drives `early_stopping_rounds=20`; the test set is never seen during training or stopping decisions.

**FastAPI lifespan over `@app.on_event`** -- the modern FastAPI pattern loads all three joblib objects once at startup into `app.state.predictor`. No per-request loading, no global variables. `/health` returns 503 -- not 200 -- when models are not loaded, which matters for load balancers and readiness probes.

**`selfMade` inferred, never requested** -- the API caller only provides observable attributes. `selfMade` is predicted by the classifier and propagated as a feature to the regressor and clusterer. `WorthResponse.selfMade_used` exposes this transparently.

**Pipeline-first, notebook-second** -- `pipeline.py` is the canonical way to produce trained artifacts. Notebooks call the same `src/` functions for exploration and visualisation.

---

## Hardware Notes

- Developed on Windows 11, NVIDIA RTX 3050 4GB VRAM
- XGBoost runs on CPU by default -- add `device="cuda"` at model construction for GPU acceleration
- Optuna HPO (40 trials x 5-fold CV) completes in ~1.5 min per model on CPU
- `OMP_NUM_THREADS=1` is set in `pipeline.py`, `conftest.py`, and `app.py` to prevent OpenMP thread-pool deadlocks on Windows

---

## License

MIT -- see [LICENSE](LICENSE).
