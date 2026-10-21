"""FastAPI service: POST /predict with a leaf photo, get the predicted disease.

Run (from the project root):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_CHECKPOINT  path of the checkpoint to serve
                      (default: checkpoints/efficientnet_b0_pretrained_last3.pt)
    CORS_ORIGINS      comma-separated allowed origins for a browser front-end
                      (default: the usual local React / Streamlit / Gradio ports)
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from PIL import UnidentifiedImageError

from src.api.schemas import HealthResponse, PredictionResponse

logger = logging.getLogger("plant-api")

DEFAULT_CHECKPOINT = "checkpoints/efficientnet_b0_pretrained_last3.pt"
DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:8501,http://localhost:7860"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # phone photos are often 5-12 MB


def load_predictor():
    """Load the model (heavy imports are done here, not at import time)."""
    checkpoint = Path(os.environ.get("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.exists():
        logger.error("Checkpoint not found: %s", checkpoint)
        return None
    from src.api.predictor import Predictor

    predictor = Predictor(checkpoint)
    logger.info("Model loaded: %s (classes: %s)", predictor.name, predictor.classes)
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = load_predictor()
    yield


app = FastAPI(
    title="Plant Disease Classifier",
    description="Classifies a leaf photo as Healthy, Powdery (powdery mildew) or Rust.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", DEFAULT_ORIGINS).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok", model_loaded=True, model_name=predictor.name, classes=predictor.classes
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded: checkpoint not found.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB).")

    try:
        return await run_in_threadpool(predictor.predict, data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file is not a valid image.")


# Website: served last, so /health, /predict and /docs keep priority.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
