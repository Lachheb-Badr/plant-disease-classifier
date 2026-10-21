# LeafScan - Plant Disease Classifier

[![CI](https://github.com/Lachheb-Badr/plant-disease-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Lachheb-Badr/plant-disease-classifier/actions/workflows/ci.yml)

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)

An end-to-end computer vision project: upload a photo of a leaf and the model tells you whether it is **healthy**, shows **powdery mildew**, or shows **rust**.

It covers the full ML lifecycle: data exploration, a reproducible data pipeline, experiment tracking, a from-scratch baseline against transfer learning, evaluation with confidence intervals, an API with a web interface, Docker, tests and CI/CD.

## Results

Both models were evaluated on a **held-out test set of 150 images** (50 per class) that was never used for training or model selection.

| Model | Trained | Test accuracy | 95% CI | Errors | Macro-F1 |
|---|---|---|---|---|---|
| Custom CNN (baseline) | from scratch, ~98k parameters | 88.0% | 81.8 - 92.3% | 18 / 150 | 0.881 |
| **EfficientNet-B0** | ImageNet weights, head + last 3 blocks fine-tuned (3.2M trainable parameters) | **98.7%** | 95.3 - 99.6% | **2 / 150** | **0.987** |

- The confidence intervals do not overlap, so the gap between the two models is not a small-sample accident.
- EfficientNet's two mistakes are both **Powdery leaves predicted as Healthy**. Rust was classified perfectly.
- The baseline mostly struggles with Rust (8 of 50 missed).
- Inference takes about 0.4 s per image on a 2-core laptop CPU.

Experiments are tracked with MLflow (parameters, per-epoch metrics, checkpoints).

## What I learned from the data (EDA)

Findings from `notebooks/01_eda.ipynb` and the decisions they led to:

| Finding | Decision |
|---|---|
| Classes are balanced in every split (train 458 / 430 / 434, validation 20 each, test 50 each) | No class weights; accuracy and macro-F1 as metrics |
| 8 different image sizes, mostly about 4000 x 2700 px | One-time downscale to 341 x 512, then a fixed model input |
| Dominant aspect ratio of 1.5 (3:2) | Keep the ratio: 224 x 336 input, no square squashing or center crop |
| Brightness differs between classes (Powdery brighter, Rust the widest spread) | Brightness jitter, no hue jitter (color is a real disease signal) |
| No corrupted files, no exact duplicates, no train/test leakage | No cleaning needed |

## Architecture

```
leaf photo --> web page (LeafScan) --> FastAPI /predict --> EfficientNet-B0 --> label + probabilities
                                             |
                          checkpoint (mounted file, not baked into the image)
```

- **Data pipeline**: `src/data/` - one-time preprocessing, augmentation (random crop at ~3:2, flips, rotation, color jitter), dataset and dataloaders, all driven by `configs/config.yaml`.
- **Models**: `src/models/` - a custom 4-block CNN with global pooling (works with any input size) and a wrapper for MobileNetV3 / EfficientNet-B0 / ResNet-50 with configurable partial fine-tuning (`--unfreeze-blocks N`).
- **Training**: `src/training/` - training loop, early stopping on validation loss, cosine schedule, MLflow logging, best-checkpoint saving.
- **Evaluation**: `src/evaluation/` - accuracy with a 95% confidence interval, per-class report, confusion matrix, misclassified-image grid, predictions CSV.
- **Serving**: `src/api/` + `frontend/` - FastAPI service that also serves the web interface. Inputs are validated (non-image, empty and oversized files are rejected), and inference uses the same preprocessing as training.

## Tech stack

PyTorch, torchvision, MLflow, DVC, FastAPI, Docker and Docker Compose, GitHub Actions, pytest.

## Run it

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Data

The images are not stored in Git. They are tracked with DVC (`data/raw.dvc`, pipeline in `dvc.yaml`). With access to the DVC remote:

```bash
dvc pull
```

Otherwise, place your own dataset in `data/raw/` with the structure `Train|Validation|Test / Healthy|Powdery|Rust /`, then build the processed images:

```bash
dvc repro            # or: make preprocess
```

### 3. Train

A GPU is recommended: on a CPU the commands work but are slow (see the design notes).

```bash
python -m src.training.train --model custom_cnn
python -m src.training.train --model efficientnet_b0 --unfreeze-blocks 3 --lr 0.0003
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://localhost:5000
```

### 4. Evaluate

```bash
python -m src.evaluation.evaluate --checkpoint checkpoints/efficientnet_b0_pretrained_last3.pt
```

Results are written to `artifacts/<checkpoint>/test/`.

### 5. Serve

Put the checkpoint in `checkpoints/`, then:

```bash
make api                           # http://localhost:8000
```

or with Docker (API + website, and an MLflow server):

```bash
docker compose up --build
```

The interactive API docs are at `http://localhost:8000/docs`. Example call:

```bash
curl -X POST http://localhost:8000/predict -F "file=@leaf.jpg"
```

```json
{
  "label": "Rust",
  "confidence": 0.97,
  "probabilities": {"Healthy": 0.01, "Powdery": 0.02, "Rust": 0.97},
  "model_name": "efficientnet_b0_pretrained_last3",
  "inference_time_ms": 420.0
}
```

### 6. Test

```bash
pytest tests/ -v
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`): on every push and pull request, runs the test suite, builds the Docker image and smoke-tests the running container.
- **CD** (`.github/workflows/cd.yml`): after a successful CI run on `main`, publishes the image to GitHub Container Registry.

## Project structure

```
plant-disease-classifier/
├── .github/workflows/      CI and CD pipelines
├── configs/config.yaml     single source of truth for all settings
├── data/                   raw and processed images (tracked by DVC)
├── frontend/               web interface (HTML, CSS, JS)
├── notebooks/              01_eda.ipynb
├── src/
│   ├── data/               preprocessing, transforms, dataset, dataloaders
│   ├── models/             custom CNN, pretrained wrappers, checkpoint loading
│   ├── training/           trainer, callbacks, entry point
│   ├── evaluation/         metrics and evaluation script
│   ├── api/                FastAPI app, predictor, schemas
│   └── utils/              config, seeding, CPU benchmark
├── tests/                  data, model, metrics, callbacks and API tests
├── Dockerfile              API + website image (CPU-only PyTorch)
├── Dockerfile.mlflow       MLflow server image
├── docker-compose.yml
├── dvc.yaml                preprocessing pipeline
└── Makefile                shortcuts (make help)
```

## Design notes

- **Measure before training.** A small benchmark script (`src/utils/benchmark.py`) times data loading and the model separately. On a 2-core CPU it showed about 3 minutes per epoch for the custom CNN and about an hour for EfficientNet-B0 fine-tuning.
- **Preprocess once.** The raw photos are about 11 megapixels; decoding them every epoch would dominate the training time, so they are downscaled once and tracked as a DVC pipeline stage.
- **Honest evaluation.** The validation set (60 images) is used for early stopping and checkpoint selection only. The test set is used once at the end, and results are reported with a 95% confidence interval.
- **The model is not baked into the Docker image.** It is mounted at runtime, so models can be swapped without a rebuild.
