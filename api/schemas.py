"""
api/schemas.py
--------------
Pydantic models for all API request and response payloads.

Design notes
------------
- PredictRequest does not include selfMade — the API infers it from
  the classifier so the caller only needs observable attributes.
- ``gender`` is constrained to ``Literal["M", "F"]`` for strict
  boundary validation.  Unknown ``category`` / ``country`` values are
  accepted and silently map to -1 via OrdinalEncoder's
  ``handle_unknown="use_encoded_value"`` — this is logged in the
  predictor but does not raise an error.
- Every response model is flat (no nested dicts) so the OpenAPI docs
  render cleanly and clients can access fields without extra unpacking.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ── Request ───────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Input features required for all prediction endpoints."""

    finalWorth: float = Field(
        ...,
        gt=0,
        description=(
            "Net worth in billion USD (e.g. 5.2 = $5.2B).  "
            "Internally converted to million USD to match the training scale."
        ),
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
        min_length=1,
        description=(
            "Wealth category — e.g. 'Technology', 'Finance', "
            "'Fashion & Retail', 'Manufacturing'.  "
            "Unrecognised values are encoded as -1 (out-of-vocabulary)."
        ),
        examples=["Technology"],
    )
    country: str = Field(
        ...,
        min_length=1,
        description=(
            "Country of residence — e.g. 'United States', 'China'.  "
            "Unrecognised values are encoded as -1 (out-of-vocabulary)."
        ),
        examples=["United States"],
    )
    # FIX L3 — constrain gender to the two values seen during training.
    # Any other string would silently encode to -1 and produce misleading
    # predictions with no error surfaced to the caller.
    gender: Literal["M", "F"] = Field(
        ...,
        description="Gender — must be 'M' or 'F'.",
        examples=["M"],
    )

    @field_validator("category", "country")
    @classmethod
    def _no_whitespace_only(cls, v: str) -> str:
        """Reject strings that are all whitespace after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field must not be blank or whitespace-only.")
        return stripped

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
    """Regressor output for net-worth prediction.

    .. warning::
        The regressor achieves R²≈0.05 on held-out data — it explains
        roughly 5% of log-worth variance.  ``worth_billion_usd`` reflects
        population-level trends (industry, geography, age) and should be
        treated as a directional estimate, not a precise point forecast.
    """

    log_worth_predicted: float = Field(
        ...,
        description="Predicted log1p(finalWorth_millions) — the model's native output.",
    )
    worth_billion_usd: float = Field(
        ...,
        description=(
            "Back-transformed prediction in billion USD (expm1(log_worth_predicted) / 1000)."
        ),
    )
    selfMade_used: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "The selfMade value used as a regressor feature — inferred from the classifier."
        ),
    )


class ClusterResponse(BaseModel):
    """Clusterer output for wealth segment assignment.

    Notes
    -----
    The clusterer was fitted on the full dataset (train + val + test) for
    exploratory segmentation.  Cluster IDs are stable within a given
    pipeline run but may shift if the pipeline is re-run with a different k.
    """

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
    artifacts: list[str] = Field(..., description="Names of artifact files successfully loaded.")


class MetricsResponse(BaseModel):
    """Stored evaluation metrics from the last pipeline run."""

    classifier: dict
    regressor: dict
    clusterer: dict
