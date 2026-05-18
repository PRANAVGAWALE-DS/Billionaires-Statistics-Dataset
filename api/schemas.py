"""
api/schemas.py
--------------
Pydantic models for all API request and response payloads.

Design notes
------------
- PredictRequest does not include selfMade — the API infers it from
  the classifier so the caller only needs observable attributes.
- Every response model is flat (no nested dicts) so the OpenAPI docs
  render cleanly and clients can access fields without extra unpacking.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request ───────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Input features required for all prediction endpoints."""

    finalWorth: float = Field(
        ...,
        gt=0,
        description="Net worth in billion USD (e.g. 5.2 = $5.2B).",
        examples=[5.2],
    )
    age: float = Field(
        ...,
        gt=0,
        lt=130,
        description="Age in years.",
        examples=[52],
    )
    category: str = Field(
        ...,
        description=(
            "Wealth category — e.g. 'Technology', 'Finance', "
            "'Fashion & Retail', 'Manufacturing'."
        ),
        examples=["Technology"],
    )
    country: str = Field(
        ...,
        description="Country of residence — e.g. 'United States', 'China'.",
        examples=["United States"],
    )
    gender: str = Field(
        ...,
        description="Gender — 'M' or 'F'.",
        examples=["M"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "finalWorth": 5.2,
                "age": 52,
                "category": "Technology",
                "country": "United States",
                "gender": "M",
            }
        }
    }


# ── Model-specific responses ──────────────────────────────────────────────


class SelfMadeResponse(BaseModel):
    """Classifier output for self-made vs inherited wealth prediction."""

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being self-made P(selfMade=1).",
    )
    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Hard prediction: 1 = Self-Made, 0 = Inherited.",
    )
    label: str = Field(
        ...,
        description="Human-readable label: 'Self-Made' or 'Inherited'.",
    )


class WorthResponse(BaseModel):
    """Regressor output for net-worth prediction."""

    log_worth_predicted: float = Field(
        ...,
        description="Predicted log1p(finalWorth) — the model's native output.",
    )
    worth_billion_usd: float = Field(
        ...,
        description=(
            "Back-transformed prediction in billion USD "
            "(expm1 of log_worth_predicted)."
        ),
    )
    selfMade_used: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "The selfMade value used as a regressor feature — "
            "inferred from the classifier."
        ),
    )


class ClusterResponse(BaseModel):
    """Clusterer output for wealth segment assignment."""

    cluster: int = Field(
        ...,
        ge=0,
        description="Assigned wealth segment cluster ID (0-indexed).",
    )
    n_clusters: int = Field(
        ...,
        gt=0,
        description="Total number of clusters the model was fitted with.",
    )
    silhouette: float = Field(
        ...,
        description="Silhouette score of the clustering (from training evaluation).",
    )


# ── Combined response ─────────────────────────────────────────────────────


class PredictResponse(BaseModel):
    """Combined response from all three models in a single call."""

    self_made: SelfMadeResponse
    worth: WorthResponse
    cluster: ClusterResponse


# ── Utility responses ─────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="'ok' when all models are loaded.")
    models_loaded: bool
    artifacts: list[str] = Field(
        ..., description="Names of artifact files successfully loaded."
    )


class MetricsResponse(BaseModel):
    """Stored evaluation metrics from the last pipeline run."""

    classifier: dict
    regressor: dict
    clusterer: dict
