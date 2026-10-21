"""Request/response models of the API."""
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    label: str = Field(description="Predicted class")
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted class")
    probabilities: dict[str, float] = Field(description="Probability of every class")
    model_name: str = Field(description="Name of the checkpoint used")
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    classes: list[str] = []
