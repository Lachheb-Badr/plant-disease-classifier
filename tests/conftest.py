"""Shared fixtures: the real config with tiny sizes, and a synthetic dataset."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.utils.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@pytest.fixture
def config():
    cfg = load_config(CONFIG_PATH)  # also checks that config.yaml stays valid
    cfg["data"]["image_size"] = [64, 96]
    cfg["data"]["preprocess_size"] = [80, 120]
    cfg["dataloader"].update(batch_size=4, num_workers=0, pin_memory=False)
    return cfg


@pytest.fixture
def fake_data(tmp_path, config):
    """Tiny Split/Class folder structure of random images; the config points to it."""
    rng = np.random.default_rng(0)
    for folder in config["data"]["splits"].values():
        for cls in config["data"]["classes"]:
            d = tmp_path / folder / cls
            d.mkdir(parents=True)
            for i in range(4):
                arr = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"img_{i}.jpg")
    config["data"]["processed_dir"] = str(tmp_path)
    return config
