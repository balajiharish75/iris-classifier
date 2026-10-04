import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("iris-classifier")

app = FastAPI(
    title="Iris Classifier API",
    description="Predicts iris flower species from sepal and petal measurements.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except FileNotFoundError:
    model = None
    logger.error("Model file not found: %s", MODEL_PATH)

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Iris features: [sepal_length, sepal_width, petal_length, petal_width]",
    )


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    request_body = ""
    if request.method == "POST":
        try:
            request_body = (await request.body()).decode("utf-8", errors="replace")
        except Exception:
            request_body = ""

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "%s %s -> 500 | %.1f ms | body=%s",
            request.method,
            request.url.path,
            duration_ms,
            request_body[:200],
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    suffix = f" | body={request_body[:200]}" if request_body else ""
    logger.info(
        "%s %s -> %s | %.1f ms%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        suffix,
    )
    return response


@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py to create model.pkl.",
        )

    features = np.array(request.features).reshape(1, -1)
    pred = int(model.predict(features)[0])
    class_name = IRIS_CLASSES[pred]

    logger.info("prediction=%d (%s) | features=%s", pred, class_name, request.features)

    return PredictionResponse(prediction=pred, class_name=class_name)
