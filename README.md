# 💰 Billionaires Statistics — Advanced Analysis Pipeline

> End-to-end data science project: EDA, statistical testing, wealth inequality
> analysis, geospatial visualisation, XGBoost ML with Optuna HPO, and SHAP
> explainability — built on the
> [Kaggle Billionaires Statistics Dataset (2023)](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset).

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

## Quickstart

```bash
# 1. Clone & enter the project
git clone https://github.com/your-username/billionaires-analysis.git
cd billionaires-analysis

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies (editable install — src/ is importable)
pip install -e .

# 4. Drop the dataset
#    Download from Kaggle and place at:
#    data/raw/Billionaires Statistics Dataset.csv

# 5. Run the notebooks
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb
```

---

## What's Inside

### `01_eda.ipynb` — Exploratory Data Analysis
| Topic | Detail |
|---|---|
| Data Quality | Missing-value audit, schema validation |
| Cleaning | Per-category median age imputation, type coercion, deduplication |
| Feature Engineering | `log_worth`, `age_group`, `wealth_per_decade`, `continent` |
| Distributions | Raw vs log(1+x) wealth, age by self-made status |
| Statistical Tests | t-test (self-made vs inherited), ANOVA (by category), Chi-squared (gender × self-made) |
| Wealth Inequality | Gini coefficient, Lorenz curve, top-1% / top-10% concentration |
| Geospatial | Plotly choropleth — count and total wealth by country |

### `02_modeling.ipynb` — Machine Learning
| Topic | Detail |
|---|---|
| Classification | Predict `selfMade` — XGBoost + Optuna (40 trials, 5-fold CV, ROC-AUC) |
| Regression | Predict `log_worth` — XGBoost + Optuna (40 trials, 5-fold CV, R²) |
| Evaluation | Confusion matrix, classification report, actual-vs-predicted scatter |
| Explainability | SHAP TreeExplainer — bar & beeswarm for both models |
| Clustering | K-Means with elbow method + PCA 2D projection |

---

## Key Design Decisions

**Thin notebooks, fat `src/`** — all reusable logic (loading, cleaning, feature
engineering, model classes, plotting) lives in the `billionaires` package.
Notebooks import and call; they never define.  This means:
- You can unit-test the source code with pytest
- You can import any function in a new experiment without copy-pasting
- The notebooks are readable narratives, not walls of code

**Log-transform the target** — `finalWorth` has extreme right skew (skewness > 10).
The regressor trains and evaluates on `log(1 + finalWorth)` to stabilise
gradients; `WorthRegressor.predict(exponentiate=True)` converts back to dollars.

**Per-group imputation** — age is imputed with the median within each `category`
group, not the global mean.  A 40-year-old tech founder and an 80-year-old
manufacturing heir are in different populations.

---

## Hardware Notes
- Developed on Windows 11, NVIDIA RTX 3050 4GB
- XGBoost runs on CPU by default; set `tree_method="hist", device="cuda"`
  in the model params to enable GPU acceleration on compatible hardware
- Optuna HPO (40 trials × 5-fold) completes in ~5 min on CPU

---

## License
MIT
