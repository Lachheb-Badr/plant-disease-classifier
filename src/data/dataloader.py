"""Build train/val/test DataLoaders from the config."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PlantDiseaseDataset
from src.data.transforms import get_eval_transforms, get_train_transforms


def get_dataloaders(config: dict) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    dl_cfg = config["dataloader"]
    root = Path(data_cfg["processed_dir"])
    classes = data_cfg["classes"]

    transforms = {
        "train": get_train_transforms(config),
        "val": get_eval_transforms(config),
        "test": get_eval_transforms(config),
    }

    generator = torch.Generator().manual_seed(config["project"]["seed"])
    loaders = {}
    for key, folder in data_cfg["splits"].items():
        dataset = PlantDiseaseDataset(root / folder, classes, transforms[key])
        loaders[key] = DataLoader(
            dataset,
            batch_size=dl_cfg["batch_size"],
            shuffle=(key == "train"),
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"] and torch.cuda.is_available(),
            generator=generator if key == "train" else None,
        )
    return loaders
