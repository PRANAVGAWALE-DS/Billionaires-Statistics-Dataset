# 💰 Billionaires Statistics — Advanced Analysis Pipeline

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-blueviolet.svg)](https://optuna.org)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-red.svg)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end data science pipeline on the Kaggle Billionaires Statistics Dataset (2023) — EDA and statistical testing, wealth inequality analysis, geospatial visualisation, XGBoost modelling with Optuna HPO, SHAP explainability, and K-Means clustering.

> **Dataset →** [Kaggle Billionaires Statistics Dataset (2023)](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset)

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Notebooks](#notebooks)
5. [ML Models](#ml-models)
6. [Analysis Highlights](#analysis-highlights)
7. [Key Design Decisions](#key-design-decisions)
8. [Hardware Notes](#hardware-notes)

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — nelgiriyewithana](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset) |
| Records | 2,640 billionaires |
| Columns | 35 raw → 43 after feature engineering |
| Key columns | `finalWorth`, `age`, `selfMade`, `category`, `country`, `gender` |

### Data Quality Summary

| Column | Issue | Handling |
|---|---|---|
| `age` | 2.46% missing | Per-category median imputation |
| `country` | 1.44% missing | Retained as-is |
| `organization` / `title` | ~87% missing | Not used in modelling |
| `selfMade` | Raw dtype `object` | Cast to `int64` in `loader.clean()` |

---

## Project Structure

```
billionaires-analysis/
├── data/
│   └── raw/                         ← place the CSV here (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb                 ← EDA, stats, inequality, maps
│   └── 02_modeling.ipynb            ← ML, SHAP, clustering
├── reports/
│   └── figures/                     ← auto-populated on notebook run
├── src/
│   └── billionaires/
│       ├── data/
│       │   └── loader.py            ← load_raw, validate_schema, clean
│       ├── features/
│       │   └── engineer.py          ← build_features, feature lists
│       ├── models/
│       │   ├── classifier.py        ← SelfMadeClassifier (XGBoost + Optuna)
│       │   └── regressor.py         ← WorthRegressor (XGBoost + Optuna)
│       └── viz/
│           └── plots.py             ← all reusable figure functions
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1 — Clone and enter
git clone https://github.com/PRANAVGAWALE-DS/Billionaires-Statistics-Dataset.git
cd billionaires-analysis

# 2 — Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3 — Install dependencies (editable — src/ is importable immediately)
pip install -e .

# 4 — Place the dataset
# Download from Kaggle and save at:
# data/raw/Billionaires Statistics Dataset.csv

# 5 — Run the notebooks
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb
```

---

## Notebooks

### `01_eda.ipynb` — Exploratory Data Analysis

| Section | Detail |
|---|---|
| Data Quality | Missing-value audit, schema validation, dtype inspection |
| Cleaning | Per-category median age imputation, type coercion, deduplication |
| Feature Engineering | `log_worth`, `age_group`, `wealth_per_decade`, `continent`, label-encoded categoricals |
| Distributions | Raw vs log(1+x) wealth · age by self-made status (violin + box) |
| Statistical Tests | t-test (self-made vs inherited) · ANOVA (by category) · Chi-squared (gender × self-made) |
| Wealth Inequality | Gini coefficient, Lorenz curve, top-1% / top-10% concentration |
| Geospatial | Plotly choropleth — billionaire count and total wealth by country |

### `02_modeling.ipynb` — Machine Learning

| Section | Detail |
|---|---|
| Classification | Predict `selfMade` — XGBoost + Optuna HPO |
| Regression | Predict `log_worth` — XGBoost + Optuna HPO |
| Evaluation | Confusion matrix · classification report · actual-vs-predicted scatter |
| Explainability | SHAP `TreeExplainer` — bar and beeswarm plots for both models |
| Clustering | K-Means (elbow method, k=4) + PCA 2D projection + cluster profile table |

---

## ML Models

### 1 · Self-Made Classifier

Predicts whether a billionaire is self-made (`1`) or inherited (`0`) from age, net worth, and geographic / industry features.

| Property | Detail |
|---|---|
| Algorithm | XGBoost binary classifier + Optuna HPO (40 trials, 5-fold stratified CV) |
| Target | `selfMade` — binary 0 / 1 |
| Split | 70% train / 15% val (early stopping, `rounds=20`) / 15% test |
| Features | `age`, `log_worth`, `wealth_per_decade`, `category_enc`, `country_enc`, `gender_enc`, `continent_enc` |
| Optimised metric | ROC-AUC |
| **ROC-AUC (test)** | **0.8436** |
| **Accuracy (test)** | **0.81** |
| Explainability | SHAP `TreeExplainer` — beeswarm + bar |

### 2 · Net Worth Regressor

Predicts `log(1 + finalWorth)` from demographic and geographic features. `wealth_per_decade` is intentionally excluded — it encodes `finalWorth` directly and would constitute target leakage.

| Property | Detail |
|---|---|
| Algorithm | XGBoost regression + Optuna HPO (40 trials, 5-fold KFold CV) |
| Target | `log_worth = log1p(finalWorth)` |
| Split | 70% train / 15% val (early stopping, `rounds=20`) / 15% test |
| Features | `age`, `selfMade`, `category_enc`, `country_enc`, `gender_enc`, `continent_enc` |
| Optimised metric | R² |
| **R² (test)** | **0.08** |
| **MAE (test)** | **0.58** |
| Explainability | SHAP `TreeExplainer` — dot plot |

> **Note on R²:** Predicting net worth magnitude from age, self-made status, and geography alone is a genuinely hard problem — these features carry limited signal about wealth scale. The low R² is honest, not a model failure. The SHAP plots reveal which features drive the predictions that are made.

---

## Analysis Highlights

### Statistical Tests

| Test | Question | Result |
|---|---|---|
| Welch's t-test | Do self-made and inherited billionaires differ in net worth? | Not significant — p = 0.22 |
| One-way ANOVA | Do mean net worths differ across the top 8 categories? | Significant — F = 4.46, p < 0.001 |
| Chi-squared | Is gender associated with self-made status? | Significant — χ² = 287.2, p < 0.001 |

### Key Findings

| Metric | Value |
|---|---|
| Dataset size (after cleaning) | 2,640 billionaires |
| Self-made share | 69.2% |
| Median net worth | $2.3B |
| Top 1% wealth share | 18.0% |
| Top 10% wealth share | 47.8% |
| Gini coefficient | > 0.6 (extreme concentration within the billionaire class itself) |
| Gender split | 85.3% male · 14.7% female |
| Female self-made rate | 28.5% vs male 72.5% — χ² significant |

---

## Key Design Decisions

**Thin notebooks, fat `src/`** — all reusable logic lives in the `billionaires` package. Notebooks import and call; they never define. This means the source code is unit-testable, any function is importable without copy-pasting, and notebooks read as clean narratives.

**Log-transform the target** — `finalWorth` has skewness > 10. The regressor trains and evaluates on `log1p(finalWorth)` to stabilise gradients. `WorthRegressor.predict(exponentiate=True)` converts predictions back to the original dollar scale.

**Per-group age imputation** — age is imputed with the within-`category` median rather than the global mean. A 40-year-old tech founder and an 80-year-old manufacturing heir belong to different age distributions.

**Leakage guard on `wealth_per_decade`** — this feature (`finalWorth / (age / 10)`) is a valid predictor for the classifier but is excluded from the regressor's feature list since its numerator is the prediction target.

**Early stopping on a dedicated val set** — both models use a 70/15/15 split. The val set drives `early_stopping_rounds=20`; the test set is never seen during training or stopping decisions.

---

## Hardware Notes

- Developed on Windows 11, NVIDIA RTX 3050 4GB VRAM
- XGBoost runs on CPU by default — add `tree_method="hist", device="cuda"` to model params for GPU acceleration
- Optuna HPO (40 trials × 5-fold CV) completes in ~5 min on CPU for both models

---

## License

MIT — see [LICENSE](LICENSE).