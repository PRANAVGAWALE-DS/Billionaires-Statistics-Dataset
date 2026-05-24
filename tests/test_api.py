"""
tests/test_api.py
-----------------
FastAPI route tests using Starlette's TestClient.

Covers
------
- GET  /health (200 when loaded, 503 when not)
- GET  /metrics (200, correct structure)
- POST /predict (200, full response contract)
- POST /predict/self-made (200, response contract)
- POST /predict/worth (200, response contract)
- POST /predict/cluster (200, response contract)
- Pydantic schema rejection (422) for invalid inputs
- 503 response when predictor startup failed
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import BillionairesPredictor

# ── Shared test payload ────────────────────────────────────────────────────

_VALID_PAYLOAD = {
    "finalWorth": 5.2,
    "age": 52,
    "category": "Technology",
    "country": "United States",
    "gender": "M",
}


# ── Client fixtures ────────────────────────────────────────────────────────
#
# Both fixtures are function-scoped (the pytest default) — NOT module-scoped.
#
# Root cause of the original 503 failures:
#   Both fixtures mutate the same `app` singleton via `app.state.predictor`.
#   With module scope, `unloaded_client` ran once and set predictor = None.
#   That mutation persisted for the rest of the module because the
#   module-scoped `client` fixture had already been created and would not
#   re-execute to restore the predictor.  Every subsequent test using
#   `client` then received a 503.
#
# Fix:
#   Function scope ensures `app.state.predictor = loaded_predictor` is
#   re-applied before every test that uses `client`.  `unloaded_client`
#   uses `yield` so that state is explicitly restored after each test that
#   needs the predictor-absent path.  TestClient construction is cheap
#   (it does not re-fit any models), so function scope adds negligible cost.


@pytest.fixture()
def client(loaded_predictor: BillionairesPredictor) -> TestClient:
    """TestClient with the app state pre-loaded with the mock predictor.

    Re-sets ``app.state.predictor`` before every test so that a prior
    ``unloaded_client`` call cannot leave the predictor as None.
    """
    app.state.predictor = loaded_predictor
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def unloaded_client(loaded_predictor: BillionairesPredictor):
    """TestClient with predictor set to None — simulates startup failure.

    Accepts ``loaded_predictor`` solely to restore ``app.state.predictor``
    in the yield teardown, preventing state bleed into tests that follow.
    """
    app.state.predictor = None
    yield TestClient(app, raise_server_exceptions=False)
    # Restore state so the next test using `client` sees a loaded predictor.
    app.state.predictor = loaded_predictor


# ── /health ────────────────────────────────────────────────────────────────


class TestHealth:
    def test_200_when_loaded(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["models_loaded"] is True
        assert isinstance(body["artifacts"], list)
        assert len(body["artifacts"]) > 0

    def test_503_when_not_loaded(self, unloaded_client: TestClient) -> None:
        r = unloaded_client.get("/health")
        assert r.status_code == 503
        body = r.json()
        assert body["models_loaded"] is False


# ── /metrics ───────────────────────────────────────────────────────────────


class TestMetrics:
    def test_200_and_structure(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert "classifier" in body
        assert "regressor" in body
        assert "clusterer" in body

    def test_503_when_not_loaded(self, unloaded_client: TestClient) -> None:
        r = unloaded_client.get("/metrics")
        assert r.status_code == 503


# ── POST /predict ──────────────────────────────────────────────────────────


class TestPredictAll:
    def test_200_and_top_level_keys(self, client: TestClient) -> None:
        r = client.post("/predict", json=_VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"self_made", "worth", "cluster"}

    def test_self_made_response_schema(self, client: TestClient) -> None:
        body = client.post("/predict", json=_VALID_PAYLOAD).json()
        sm = body["self_made"]
        assert set(sm) == {"probability", "prediction", "label"}
        assert 0.0 <= sm["probability"] <= 1.0
        assert sm["prediction"] in (0, 1)
        assert sm["label"] in ("Self-Made", "Inherited")

    def test_worth_response_schema(self, client: TestClient) -> None:
        body = client.post("/predict", json=_VALID_PAYLOAD).json()
        worth = body["worth"]
        assert "log_worth_predicted" in worth
        assert "worth_billion_usd" in worth
        assert "selfMade_used" in worth
        assert worth["worth_billion_usd"] > 0

    def test_cluster_response_schema(self, client: TestClient) -> None:
        body = client.post("/predict", json=_VALID_PAYLOAD).json()
        cluster = body["cluster"]
        assert "cluster" in cluster
        assert "n_clusters" in cluster
        assert "silhouette" in cluster
        assert 0 <= cluster["cluster"] < cluster["n_clusters"]

    def test_503_when_not_loaded(self, unloaded_client: TestClient) -> None:
        r = unloaded_client.post("/predict", json=_VALID_PAYLOAD)
        assert r.status_code == 503


# ── POST /predict/self-made ────────────────────────────────────────────────


class TestPredictSelfMade:
    def test_200_and_schema(self, client: TestClient) -> None:
        r = client.post("/predict/self-made", json=_VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"probability", "prediction", "label"}
        assert 0.0 <= body["probability"] <= 1.0

    def test_503_when_not_loaded(self, unloaded_client: TestClient) -> None:
        r = unloaded_client.post("/predict/self-made", json=_VALID_PAYLOAD)
        assert r.status_code == 503


# ── POST /predict/worth ────────────────────────────────────────────────────


class TestPredictWorth:
    def test_200_and_schema(self, client: TestClient) -> None:
        r = client.post("/predict/worth", json=_VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert "worth_billion_usd" in body
        assert body["worth_billion_usd"] > 0

    def test_503_when_not_loaded(self, unloaded_client: TestClient) -> None:
        r = unloaded_client.post("/predict/worth", json=_VALID_PAYLOAD)
        assert r.status_code == 503


# ── POST /predict/cluster ──────────────────────────────────────────────────


class TestPredictCluster:
    def test_200_and_schema(self, client: TestClient) -> None:
        r = client.post("/predict/cluster", json=_VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert "cluster" in body
        assert body["cluster"] >= 0

    def test_503_when_not_loaded(self, unloaded_client: TestClient) -> None:
        r = unloaded_client.post("/predict/cluster", json=_VALID_PAYLOAD)
        assert r.status_code == 503


# ── Schema validation (422) ────────────────────────────────────────────────


class TestSchemaValidation:
    def test_missing_field_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "finalWorth"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_negative_worth_returns_422(self, client: TestClient) -> None:
        r = client.post("/predict", json={**_VALID_PAYLOAD, "finalWorth": -1.0})
        assert r.status_code == 422

    def test_zero_worth_returns_422(self, client: TestClient) -> None:
        r = client.post("/predict", json={**_VALID_PAYLOAD, "finalWorth": 0.0})
        assert r.status_code == 422

    def test_invalid_gender_returns_422(self, client: TestClient) -> None:
        """FIX L3 — gender must be 'M' or 'F'; any other value is rejected."""
        r = client.post("/predict", json={**_VALID_PAYLOAD, "gender": "X"})
        assert r.status_code == 422

    def test_blank_category_returns_422(self, client: TestClient) -> None:
        r = client.post("/predict", json={**_VALID_PAYLOAD, "category": "   "})
        assert r.status_code == 422

    def test_age_out_of_range_returns_422(self, client: TestClient) -> None:
        r = client.post("/predict", json={**_VALID_PAYLOAD, "age": 200})
        assert r.status_code == 422

    def test_unknown_country_accepted(self, client: TestClient) -> None:
        """Unknown country is valid input — OrdinalEncoder maps it to -1."""
        r = client.post("/predict", json={**_VALID_PAYLOAD, "country": "Wakanda"})
        assert r.status_code == 200

    def test_unknown_category_accepted(self, client: TestClient) -> None:
        """Unknown category is valid input — OrdinalEncoder maps it to -1."""
        r = client.post("/predict", json={**_VALID_PAYLOAD, "category": "Astrology"})
        assert r.status_code == 200
