"""
api/main.py
-----------
FastAPI application for the Billionaires ML pipeline.

Start the server
----------------
::

    # from the project root (billionaires-analysis/)
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

    # or with the helper script
    python -m api.main

Interactive docs
----------------
    http://localhost:8000/docs     ← Swagger UI
    http://localhost:8000/redoc    ← ReDoc

Endpoints
---------
GET  /health                  → model load status and artifact list
GET  /metrics                 → stored evaluation metrics (from metrics.json)
POST /predict                 → run all three models; returns combined response
POST /predict/self-made       → classifier only  (P(selfMade))
POST /predict/worth           → regressor only   (log_worth + back-transform)
POST /predict/cluster         → clusterer only   (wealth segment)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.predictor import BillionairesPredictor
from api.schemas import (
    ClusterResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    SelfMadeResponse,
    WorthResponse,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# ── Processed-dir override via env var ────────────────────────────────────
_PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "data/processed"))


# ── Lifespan — models loaded once at startup ──────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all serialised artifacts before the first request is served.

    Using the FastAPI lifespan context manager (the modern replacement for
    ``@app.on_event("startup")``) means models are loaded exactly once,
    are available in ``app.state.predictor``, and are never reloaded per
    request.
    """
    logger.info("Loading artifacts from '%s' ...", _PROCESSED_DIR)
    predictor = BillionairesPredictor(processed_dir=_PROCESSED_DIR)
    try:
        predictor.load()
        logger.info("All artifacts loaded. API is ready.")
    except FileNotFoundError as exc:
        logger.error("Startup failed: %s", exc)
        # Store the error; /health will expose it cleanly.
        predictor = None  # type: ignore[assignment]

    app.state.predictor = predictor
    yield
    # No teardown needed — joblib objects are in-memory only.


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Billionaires Statistics — ML API",
    description=(
        "Serves three XGBoost models trained on the Kaggle Billionaires "
        "Statistics Dataset (2023): a self-made classifier, a net-worth "
        "regressor, and a wealth-segment clusterer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Shared dependency helper ──────────────────────────────────────────────


def _get_predictor(request: Request) -> BillionairesPredictor:
    """Return the loaded predictor or raise 503 if startup failed."""
    predictor: BillionairesPredictor | None = request.app.state.predictor
    if predictor is None or not predictor.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Models are not loaded. "
                "Run `python pipeline.py --cluster-k 4` to generate artifacts, "
                "then restart the server."
            ),
        )
    return predictor


# ── Utility endpoints ─────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Utility"],
)
def health(request: Request) -> HealthResponse:
    """Return 200 when all models are loaded, 503 otherwise.

    Unlike naive health endpoints that always return 200, this one
    reflects the actual model load status.
    """
    predictor: BillionairesPredictor | None = request.app.state.predictor
    if predictor is None or not predictor.loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "models_loaded": False,
                "artifacts": [],
            },
        )
    return HealthResponse(
        status="ok",
        models_loaded=True,
        artifacts=predictor.artifact_names,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Stored evaluation metrics",
    tags=["Utility"],
)
def metrics(request: Request) -> MetricsResponse:
    """Return the evaluation metrics written to ``metrics.json``
    during the last ``pipeline.py`` run."""
    predictor = _get_predictor(request)
    m = predictor.metrics
    return MetricsResponse(
        classifier=m.get("classifier", {}),
        regressor=m.get("regressor", {}),
        clusterer=m.get("clusterer", {}),
    )


# ── Prediction endpoints ──────────────────────────────────────────────────


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run all three models",
    tags=["Prediction"],
)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Run the classifier, regressor, and clusterer in a single call.

    ``selfMade`` is inferred from the classifier and propagated as a
    feature to the regressor and clusterer — callers never need to
    supply it.
    """
    predictor = _get_predictor(request)
    result = predictor.predict_all(
        finalWorth=body.finalWorth,
        age=body.age,
        category=body.category,
        country=body.country,
        gender=body.gender,
    )
    return PredictResponse(
        self_made=SelfMadeResponse(**result["self_made"]),
        worth=WorthResponse(**result["worth"]),
        cluster=ClusterResponse(**result["cluster"]),
    )


@app.post(
    "/predict/self-made",
    response_model=SelfMadeResponse,
    summary="Self-made classifier",
    tags=["Prediction"],
)
def predict_self_made(body: PredictRequest, request: Request) -> SelfMadeResponse:
    """Predict whether the billionaire is self-made or inherited.

    Returns the probability P(selfMade=1), hard prediction, and
    a human-readable label.
    """
    predictor = _get_predictor(request)
    result = predictor.predict_self_made(
        finalWorth=body.finalWorth,
        age=body.age,
        category=body.category,
        country=body.country,
        gender=body.gender,
    )
    return SelfMadeResponse(**result)


@app.post(
    "/predict/worth",
    response_model=WorthResponse,
    summary="Net-worth regressor",
    tags=["Prediction"],
)
def predict_worth(body: PredictRequest, request: Request) -> WorthResponse:
    """Predict net worth in log scale and back-transform to billion USD.

    ``selfMade`` is inferred from the classifier and noted in the
    response as ``selfMade_used`` for transparency.
    """
    predictor = _get_predictor(request)
    sm = predictor.predict_self_made(
        finalWorth=body.finalWorth,
        age=body.age,
        category=body.category,
        country=body.country,
        gender=body.gender,
    )
    result = predictor.predict_worth(
        finalWorth=body.finalWorth,
        age=body.age,
        category=body.category,
        country=body.country,
        gender=body.gender,
        selfMade=sm["prediction"],
    )
    return WorthResponse(**result)


@app.post(
    "/predict/cluster",
    response_model=ClusterResponse,
    summary="Wealth-segment clusterer",
    tags=["Prediction"],
)
def predict_cluster(body: PredictRequest, request: Request) -> ClusterResponse:
    """Assign the billionaire to a wealth segment cluster (0 to k-1).

    ``selfMade`` is inferred from the classifier before cluster
    assignment.
    """
    predictor = _get_predictor(request)
    sm = predictor.predict_self_made(
        finalWorth=body.finalWorth,
        age=body.age,
        category=body.category,
        country=body.country,
        gender=body.gender,
    )
    result = predictor.predict_cluster(
        finalWorth=body.finalWorth,
        age=body.age,
        category=body.category,
        country=body.country,
        gender=body.gender,
        selfMade=sm["prediction"],
    )
    return ClusterResponse(**result)


# ── Dev entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
