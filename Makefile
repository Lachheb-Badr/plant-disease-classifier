.PHONY: help install train evaluate api mlflow docker-build docker-up docker-down test dvc-push clean

# Default target
help:
	@echo ""
	@echo "🌿 Plant Disease Classifier - Available Commands"
	@echo "================================================"
	@echo "  make install        Install dependencies"
	@echo "  make train          Run training pipeline"
	@echo "  make evaluate       Run evaluation on test set"
	@echo "  make api            Start FastAPI server"
	@echo "  make mlflow         Start MLflow tracking server"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start all services via Docker Compose"
	@echo "  make docker-down    Stop all services"
	@echo "  make test           Run test suite"
	@echo "  make dvc-push       Push data to DVC remote"
	@echo "  make clean          Remove cache and temporary files"
	@echo ""

install:
	pip install -r requirements.txt

train:
	python src/training/train.py --config configs/config.yaml

evaluate:
	python src/evaluation/evaluate.py --config configs/config.yaml

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow server --host 0.0.0.0 --port 5000

docker-build:
	docker build -t plant-disease-classifier .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

dvc-push:
	dvc push

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -delete
