"""Model wrapper used by the API."""
import io
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.data.transforms import get_eval_transforms
from src.models.checkpoint import load_checkpoint


class Predictor:
    """Loads a checkpoint once and classifies images."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.model, config = load_checkpoint(self.checkpoint_path, self.device)
        self.classes = config["data"]["classes"]
        self.transform = get_eval_transforms(config)
        # Training images were first downscaled to `preprocess_size`; do the same
        # here so inference sees the same kind of image as training.
        self.preprocess_hw = tuple(config["data"]["preprocess_size"])
        self.name = self.checkpoint_path.stem

    def _prepare(self, image_bytes: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            height, width = self.preprocess_hw
            im = im.resize((width, height), Image.LANCZOS)
        return self.transform(im).unsqueeze(0)

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> dict:
        start = time.perf_counter()
        x = self._prepare(image_bytes).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "label": self.classes[best],
            "confidence": round(probs[best], 4),
            "probabilities": {c: round(p, 4) for c, p in zip(self.classes, probs)},
            "model_name": self.name,
            "inference_time_ms": round((time.perf_counter() - start) * 1000, 1),
        }
