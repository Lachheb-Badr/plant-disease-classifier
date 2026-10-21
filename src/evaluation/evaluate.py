"""Evaluate a trained checkpoint on one split (default: the test set).

Usage (from the project root):
    python -m src.evaluation.evaluate --checkpoint checkpoints/efficientnet_b0_pretrained_last3.pt
    python -m src.evaluation.evaluate --checkpoint checkpoints/custom_cnn.pt --mlflow

Outputs go to artifacts/<checkpoint name>/<split>/:
    metrics.json, classification_report.txt, predictions.csv,
    confusion_matrix.png, misclassified.png
"""
import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PlantDiseaseDataset
from src.data.transforms import get_eval_transforms
from src.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_misclassified,
    text_report,
)
from src.models.checkpoint import load_checkpoint


@torch.no_grad()
def predict(model, loader, device):
    preds, confs, targets = [], [], []
    for images, labels in tqdm(loader, desc="Evaluating"):
        probs = torch.softmax(model(images.to(device)), dim=1)
        conf, pred = probs.max(dim=1)
        preds += pred.cpu().tolist()
        confs += conf.cpu().tolist()
        targets += labels.tolist()
    return preds, confs, targets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--mlflow", action="store_true", help="Also log the results to MLflow")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    classes = config["data"]["classes"]

    split_dir = Path(config["data"]["processed_dir"]) / config["data"]["splits"][args.split]
    dataset = PlantDiseaseDataset(split_dir, classes, get_eval_transforms(config))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Checkpoint: {args.checkpoint} | split: {args.split} ({len(dataset)} images) | device: {device}")

    preds, confs, targets = predict(model, loader, device)

    out_dir = Path(args.output_dir) / Path(args.checkpoint).stem / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(targets, preds, classes)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = text_report(targets, preds, classes)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(targets, preds, classes, out_dir / "confusion_matrix.png",
                          title=f"{Path(args.checkpoint).stem} - {args.split}")

    errors = []
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true", "pred", "confidence", "correct"])
        for (path, _), t, pr, c in zip(dataset.samples, targets, preds, confs):
            writer.writerow([str(path), classes[t], classes[pr], f"{c:.4f}", t == pr])
            if t != pr:
                errors.append({"path": path, "true": classes[t], "pred": classes[pr], "conf": c})
    errors.sort(key=lambda e: -e["conf"])  # most confident mistakes first
    has_errors = plot_misclassified(errors, out_dir / "misclassified.png")

    low, high = metrics["accuracy_ci95"]
    print("\n" + report)
    print(f"Accuracy: {metrics['accuracy']:.1%} ({metrics['n_errors']} errors out of {metrics['n_samples']}) | "
          f"95% CI: {low:.1%} - {high:.1%} | macro-F1: {metrics['macro_f1']:.3f}")
    print(f"Results saved in {out_dir}" + ("" if has_errors else " (no misclassified images)"))

    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]))
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name=f"eval_{Path(args.checkpoint).stem}_{args.split}"):
            mlflow.set_tag("stage", "evaluation")
            mlflow.log_param("checkpoint", Path(args.checkpoint).name)
            mlflow.log_param("split", args.split)
            mlflow.log_metrics({
                f"{args.split}_accuracy": metrics["accuracy"],
                f"{args.split}_macro_f1": metrics["macro_f1"],
                f"{args.split}_errors": metrics["n_errors"],
            })
            mlflow.log_artifacts(str(out_dir))


if __name__ == "__main__":
    main()