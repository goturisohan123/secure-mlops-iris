import hashlib
import json
import math
import os
import secrets
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

MODEL_TEXT = Path("models/model.json").read_text()
MODEL = json.loads(MODEL_TEXT)
VERSION = hashlib.sha256(MODEL_TEXT.encode()).hexdigest()[:12]

LOCK = threading.Lock()
RECENT_INPUTS = deque(maxlen=200)
LATENCIES = deque(maxlen=200)
COUNTS = {"predictions": 0, "auth_failures": 0}

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)

Measurement = Annotated[
    float,
    Field(gt=0, le=20, allow_inf_nan=False),
]


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[Measurement] = Field(
        min_length=4,
        max_length=4,
    )


def require_key(key: str | None = Security(api_key_header)):
    expected = os.environ.get("API_KEY", "")

    valid = bool(expected and key) and secrets.compare_digest(
        (key or "").encode(),
        expected.encode(),
    )

    if not valid:
        with LOCK:
            COUNTS["auth_failures"] += 1

        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


def infer(features):
    node = 0

    while MODEL["children_left"][node] != -1:
        feature_index = MODEL["feature"][node]

        if features[feature_index] <= MODEL["threshold"][node]:
            node = MODEL["children_left"][node]
        else:
            node = MODEL["children_right"][node]

    probabilities = MODEL["probabilities"][node]

    predicted = max(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
    )

    return predicted, probabilities


@asynccontextmanager
async def lifespan(app):
    if len(os.environ.get("API_KEY", "")) < 32:
        raise RuntimeError(
            "Set API_KEY to a random value of at least 32 characters"
        )
    yield


app = FastAPI(
    title="Secure Iris MLOps API",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": VERSION,
    }


@app.post("/predict", dependencies=[Depends(require_key)])
def predict(payload: PredictionInput):
    start = time.perf_counter()

    predicted, probabilities = infer(payload.features)

    elapsed = (time.perf_counter() - start) * 1000

    with LOCK:
        COUNTS["predictions"] += 1
        RECENT_INPUTS.append(payload.features)
        LATENCIES.append(elapsed)

    return {
        "species": MODEL["class_names"][predicted],
        "class_index": predicted,
        "probabilities": probabilities,
        "model_version": VERSION,
    }


@app.get("/stats", dependencies=[Depends(require_key)])
def stats():
    with LOCK:
        latencies = sorted(LATENCIES)

        p95 = (
            latencies[math.ceil(0.95 * len(latencies)) - 1]
            if latencies
            else None
        )

        return {
            **COUNTS,
            "model_version": VERSION,
            "recent_prediction_compute_p95_ms": p95,
            "recent_inputs": list(RECENT_INPUTS),
        }
