FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Dependencies first (cached unless requirements change).
#    CPU-only PyTorch: much smaller than the default CUDA build.
COPY requirements-api.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt

# 2. Application code (the model checkpoint is mounted at runtime, not baked in).
COPY src/ src/
COPY frontend/ frontend/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV MODEL_CHECKPOINT=checkpoints/efficientnet_b0_pretrained_last3.pt
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
