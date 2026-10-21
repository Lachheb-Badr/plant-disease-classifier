"""Training loop with MLflow metric logging."""
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.training.callbacks import EarlyStopping


class Trainer:
    def __init__(self, model, loaders, config, device, run_name):
        self.model = model.to(device)
        self.loaders = loaders
        self.config = config
        self.device = device
        self.run_name = run_name

        tcfg = config["training"]
        self.epochs = tcfg["epochs"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer(tcfg)
        self.scheduler = self._build_scheduler(tcfg)

        es = tcfg["early_stopping"]
        self.early_stopping = EarlyStopping(es["patience"], es["min_delta"], es["enabled"])

        ckpt_dir = Path(config["paths"]["checkpoints_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = ckpt_dir / f"{run_name}.pt"

    # ------------------------------------------------------------------ setup
    def _build_optimizer(self, tcfg):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if tcfg["optimizer"] == "adam":
            return torch.optim.Adam(params, lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
        if tcfg["optimizer"] == "sgd":
            return torch.optim.SGD(
                params, lr=tcfg["learning_rate"], momentum=0.9, weight_decay=tcfg["weight_decay"]
            )
        raise ValueError(f"Unknown optimizer: {tcfg['optimizer']}")

    def _build_scheduler(self, tcfg):
        name = tcfg["scheduler"]
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tcfg["epochs"])
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        if name == "none":
            return None
        raise ValueError(f"Unknown scheduler: {name}")

    # --------------------------------------------------------------- one epoch
    def _run_epoch(self, loader, train: bool, desc: str):
        self.model.train(train)
        total_loss, preds, targets = 0.0, [], []
        with torch.set_grad_enabled(train):
            for images, labels in tqdm(loader, desc=desc, leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * images.size(0)
                preds.extend(outputs.argmax(dim=1).cpu().tolist())
                targets.extend(labels.cpu().tolist())
        n = len(targets)
        acc = sum(p == t for p, t in zip(preds, targets)) / n
        f1 = f1_score(targets, preds, average="macro")
        return total_loss / n, acc, f1

    # ------------------------------------------------------------------- fit
    def _save_checkpoint(self, epoch: int, val_loss: float):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "classes": self.config["data"]["classes"],
                "config": self.config,
            },
            self.checkpoint_path,
        )

    def fit(self) -> dict:
        best = {"epoch": 0, "val_loss": float("inf"), "val_acc": 0.0, "val_f1": 0.0}
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            tr_loss, tr_acc, tr_f1 = self._run_epoch(
                self.loaders["train"], True, f"Epoch {epoch}/{self.epochs} [train]"
            )
            va_loss, va_acc, va_f1 = self._run_epoch(
                self.loaders["val"], False, f"Epoch {epoch}/{self.epochs} [val]"
            )
            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            mlflow.log_metrics(
                {
                    "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                    "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
                    "lr": lr,
                },
                step=epoch,
            )

            improved = self.early_stopping.step(va_loss)
            if improved:
                best = {"epoch": epoch, "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1}
                self._save_checkpoint(epoch, va_loss)

            print(
                f"Epoch {epoch:02d}/{self.epochs} | {time.time() - start:5.0f}s | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}"
                f"{'  * best' if improved else ''}"
            )
            if self.early_stopping.should_stop:
                print(f"Early stopping: no improvement for {self.early_stopping.patience} epochs.")
                break

        # restore the best weights
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        return best
