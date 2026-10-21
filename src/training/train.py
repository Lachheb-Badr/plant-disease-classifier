"""Training entry point.

Examples (from the project root):
    python -m src.training.train --epochs 1                     # quick timing test
    python -m src.training.train --model custom_cnn
    python -m src.training.train --model efficientnet_b0 --unfreeze-blocks 3
    python -m src.training.train --model efficientnet_b0

MLflow address: the MLFLOW_TRACKING_URI environment variable if set, otherwise
`mlflow.tracking_uri` from the config (a local SQLite file by default).
"""
import argparse
import os

import mlflow
import mlflow.pytorch
import torch

from src.data.dataloader import get_dataloaders
from src.models import get_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a plant disease classifier")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", choices=["custom_cnn", "mobilenet_v3_small", "efficientnet_b0", "resnet50"])
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None,
                   help="Use ImageNet weights (default: yes for pretrained architectures)")
    p.add_argument("--unfreeze-blocks", type=int,
                   help="Partial fine-tuning: train the head + the last N backbone blocks (0 = head only)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int, help="DataLoader worker processes")
    p.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"),
                   help="Model input size, e.g. --image-size 160 240")
    p.add_argument("--run-name")
    return p.parse_args()


def blocks_suffix(name: str, mcfg: dict) -> str:
    ub = mcfg.get("unfreeze_blocks")
    if name == "custom_cnn" or ub is None:
        return ""
    return "_head_only" if ub == 0 else f"_last{ub}"


def main():
    args = parse_args()
    config = load_config(args.config)

    # command-line overrides
    if args.model:
        config["model"]["name"] = args.model
    name = config["model"]["name"]
    pretrained = args.pretrained if args.pretrained is not None else name != "custom_cnn"
    config["model"]["pretrained"] = pretrained and name != "custom_cnn"
    if args.unfreeze_blocks is not None:
        config["model"]["unfreeze_blocks"] = args.unfreeze_blocks
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["dataloader"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["dataloader"]["num_workers"] = args.num_workers
    if args.image_size:
        config["data"]["image_size"] = args.image_size

    mcfg, tcfg = config["model"], config["training"]
    run_name = args.run_name or (
        name
        + ("_pretrained" if mcfg["pretrained"] else "")
        + blocks_suffix(name, mcfg)
        + (f"_{args.image_size[0]}x{args.image_size[1]}" if args.image_size else "")
    )

    set_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run: {run_name} | device: {device}")

    loaders = get_dataloaders(config)
    model = get_model(config)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": name,
            "pretrained": mcfg["pretrained"],
            "unfreeze_blocks": ("n/a" if name == "custom_cnn" else
                                "all" if mcfg.get("unfreeze_blocks") is None else mcfg["unfreeze_blocks"]),
            "dropout": mcfg["dropout"],
            "optimizer": tcfg["optimizer"],
            "learning_rate": tcfg["learning_rate"],
            "weight_decay": tcfg["weight_decay"],
            "scheduler": tcfg["scheduler"],
            "epochs": tcfg["epochs"],
            "batch_size": config["dataloader"]["batch_size"],
            "image_size": "x".join(map(str, config["data"]["image_size"])),
            "brightness_jitter": config["transforms"]["augmentation"]["color_jitter"]["brightness"],
            "seed": config["project"]["seed"],
            "device": str(device),
            "total_params": model.num_params,
            "trainable_params": model.num_trainable_params,
        })
        mlflow.log_artifact(args.config)

        trainer = Trainer(model, loaders, config, device, run_name)
        best = trainer.fit()

        mlflow.log_metrics({
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "best_val_acc": best["val_acc"],
            "best_val_f1": best["val_f1"],
        })
        mlflow.log_artifact(str(trainer.checkpoint_path), artifact_path="checkpoints")
        try:
            # recent MLflow versions require an input example to log a PyTorch model
            example = torch.zeros(1, 3, *config["data"]["image_size"]).numpy()
            mlflow.pytorch.log_model(trainer.model.cpu().eval(), name="model", input_example=example)
        except Exception as e:  # the checkpoint is already logged: don't lose the run
            print(f"WARNING: could not log the MLflow model: {e}")

    print(f"Best epoch {best['epoch']} | val loss {best['val_loss']:.4f} | val acc {best['val_acc']:.3f}")
    print(f"Checkpoint: {trainer.checkpoint_path}")


if __name__ == "__main__":
    main()
